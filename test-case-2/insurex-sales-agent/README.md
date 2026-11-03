# InsureX Sales Assistant

An AI agent for InsureX sales staff and customers, built with **LangChain + LangGraph** and **ChromaDB**.

* **Answers questions** about 5 life-insurance products from the official brochures (`data/pdfs/`), in Thai or English, with the source page for every answer. If the brochures don't contain the answer, it says so instead of guessing.
* **Collects leads**: when a customer says they are interested, it asks for name, occupation, monthly income and phone, checks them, asks for PDPA consent, and saves the lead to SQLite through an **MCP (Model Context Protocol) tool**.
* **Keeps each customer's conversation separate**, and remembers it after a restart.

Full design notes, evaluation details and known issues: [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Example conversations

Two customers from a real run with gpt-5-mini (3 Nov 2026), copied word for word from [`logs/demo_openai-gpt-5-mini.log`](logs/demo_openai-gpt-5-mini.log). In the log the two sessions are interleaved turn by turn; here each is shown on its own. Notes in *italics* are the LangGraph path from the log, not part of the chat.

### Somchai (Thai): questions → lead → restart

> **Customer:** สวัสดีครับ
>
> **Agent:** สวัสดีครับ ผมเป็นผู้ช่วยขายของ InsureX ผมช่วยอธิบายและเปรียบเทียบแผนประกันชีวิต (คุ้มชีวา 90/10, คุ้มชีวี เอ็กซ์ตร้า 99/10, 99/15, 99/20, ตลอดชีพ 99/99, ตลอดชีพ 90/20, คุ้มมั่นใจชัวร์ แบบชั่วระยะเวลา (5/5, 10/10, 15/15)) และประสานให้ตัวแทนติดต่อกลับได้ครับ สนใจสอบถามเรื่องไหนดีครับ

> **Customer:** ตลอดชีพ 90/20 ผู้หญิงอายุ 35 ทุน 500,000 ต้องจ่ายเบี้ยปีละเท่าไหร่ครับ
>
> **Agent:** เบี้ยประกันตัวอย่างสำหรับแผน ตลอดชีพ 90/20: เพศหญิง อายุ 35 ปี ทุนประกัน 500,000 บาท เบี้ยประกันรายปี = 10,650 บาท [1]
> - ข้อมูลเป็นตัวอย่างในโบรชัวร์สำหรับเพศหญิง อายุ 35 ปี ทุนประกัน 500,000 บาท [1]
> - หากต้องการใบเสนอราคาที่แน่นอนสำหรับสถานะส่วนตัว (อายุอื่น เพศอื่น ขั้นอาชีพ หรือความคุ้มครองอื่น) กรุณาติดต่อเจ้าหน้าที่หรือตัวแทนเพื่อขอใบเสนอราคาโดยตรงครับ [1]
>
> แหล่งข้อมูล:<br>
> [1] ตลอดชีพ 90/20 – ตัวอย่างเบี้ยประกันภัยรายปี (ไฟล์ whole_life_90_20.pdf หน้า 2)

> **Customer:** แล้วแผนนี้รับประกันอายุถึงเท่าไหร่ จ่ายรายเดือนได้ไหม
>
> **Agent:** สั้นๆ: แผน ตลอดชีพ 90/20 ให้ความคุ้มครองถึงอายุ 90 ปี และสามารถชำระเบี้ยแบบรายเดือนได้ครับ [1] [2]
>
> รายละเอียดสำคัญ:
> - อายุรับประกันภัย (อายุเข้าได้): ตั้งแต่ 1 เดือน 1 วัน ถึง 65 ปี [1]
> - ระยะเวลาคุ้มครอง (สิ้นสุดความคุ้มครอง): ถึงอายุ 90 ปี [1] [2]
> - ระยะเวลาชำระเบี้ยประกันภัย: 20 ปี (แบบชำระเต็ม 20 ปี) [1] [2]
> - งวดการชำระเบี้ย: สามารถจ่ายเป็น รายปี / ราย 6 เดือน / ราย 3 เดือน / รายเดือน ได้ [1]
>
> หากต้องการเบี้ยประกันตัวอย่างหรือใบเสนอราคาที่แน่นอนสำหรับสถานะของคุณ (อายุเพศ ขั้นอาชีพ จำนวนเงินเอาประกันภัย ฯลฯ) ผมช่วยประสานตัวแทนให้ติดต่อกลับได้ครับครับ [2]
>
> แหล่งข้อมูล:<br>
> [1] ตลอดชีพ 90/20 – เงื่อนไขการรับประกันภัย (ไฟล์ whole_life_90_20.pdf หน้า 2)<br>
> [2] ตลอดชีพ 90/20 – ภาพรวมและจุดเด่นของแบบประกัน (ไฟล์ whole_life_90_20.pdf หน้า 1)

> **Customer:** เปรียบเทียบกับตลอดชีพ 99/99 หน่อย ต่างกันยังไง
>
> **Agent:** สั้นๆ: ความต่างสำคัญคือ ตลอดชีพ 90/20 ให้ความคุ้มครองถึงอายุ 90 ปี และชำระเบี้ย 20 ปี (รับเงินครบ 100% ของทุนประกันเมื่อครบสัญญา) ขณะที่ ตลอดชีพ 99/99 ให้ความคุ้มครองถึงอายุ 99 ปี และชำระเบี้ยจนถึงอายุ 99 ปี (รับเงินครบ 101% ของทุนประกันเมื่อครบสัญญา) ครับ [1] [2] [3]
>
> แหล่งข้อมูล:<br>
> [1] ตลอดชีพ 90/20 – ข้อมูลสรุปผลิตภัณฑ์ (ไฟล์ whole_life_90_20.pdf หน้า 1)<br>
> [2] ตลอดชีพ 90/20 – ผลประโยชน์ความคุ้มครอง (ไฟล์ whole_life_90_20.pdf หน้า 2)<br>
> [3] ตลอดชีพ 99/99 – ภาพรวมและจุดเด่นของแบบประกัน (ไฟล์ whole_life_99_99.pdf หน้า 1)

*Both products are retrieved for one comparison question (`grade_documents:4/4`).*

> **Customer:** สนใจ 90/20 ครับ อยากให้ตัวแทนติดต่อกลับ
>
> **Agent:** ยินดีเลยครับ เพื่อให้ตัวแทนของ InsureX ติดต่อกลับไปให้ข้อมูลแผน ตลอดชีพ 90/20 และนำเสนอแผนที่เหมาะกับคุณ รบกวนขอข้อมูลต่อไปนี้ครับ: ชื่อ-นามสกุล, อาชีพ, รายได้ต่อเดือน, เบอร์โทรศัพท์ติดต่อ

> **Customer:** ผมชื่อ สมชาย ใจดี อาชีพ วิศวกร แล้วแผนนี้ลดหย่อนภาษีได้ไหม
>
> **Agent:** ได้ครับ แผน ตลอดชีพ 90/20 สามารถนำเบี้ยประกันไปลดหย่อนภาษีได้ ตามเงื่อนไขกรมสรรพากร (ระบุในเอกสารว่าลดหย่อนภาษีได้สูงสุด 100,000 บาท) [1] [2]
> - ข้อความจากภาพรวม: ระบุว่าลดหย่อนภาษีสูงสุด 100,000 บาท ตามเงื่อนไขกรมสรรพากร [1]
> - หมายเหตุ: ระบุว่าเบี้ยประกันภัยสามารถนำไปลดหย่อนภาษีเงินได้บุคคลธรรมดาตามเกณฑ์ที่กรมสรรพากรกำหนด (รายละเอียดเกณฑ์เป็นไปตามกรมสรรพากร) [2]
>
> หากต้องการให้ตัวแทนติดต่อเพื่อตรวจสอบสิทธิ์ลดหย่อนสำหรับสถานะของคุณโดยละเอียด ผมช่วยประสานตัวแทนให้ติดต่อกลับได้ครับ [2]
>
> แหล่งข้อมูล:<br>
> [1] ตลอดชีพ 90/20 – ภาพรวมและจุดเด่นของแบบประกัน (ไฟล์ whole_life_90_20.pdf หน้า 1)<br>
> [2] ตลอดชีพ 90/20 – หมายเหตุ ข้อกำหนด และข้อมูลบริษัท (ไฟล์ whole_life_90_20.pdf หน้า 2)
>
> ขอบคุณครับ คุณสมชาย ใจดี รบกวนขอรายได้ต่อเดือน, เบอร์โทรศัพท์ติดต่อเพิ่มเติมด้วยครับ

*One message, two jobs: the name and occupation go into the lead, and the tax question is answered from the brochure. The first draft failed the number check, so the answer was regenerated (cycle 2: `verify_answer:fail → generate#2 → verify_answer:ok`).*

> **Customer:** เงินเดือน 5 หมื่นครับ เบอร์ 08123
>
> **Agent:** ขออภัยครับ เบอร์โทรศัพท์ติดต่อ ที่ให้มายังไม่ถูกต้องหรือไม่ชัดเจน ขอบคุณครับ คุณสมชาย ใจดี รบกวนขอเบอร์โทรศัพท์ติดต่อเพิ่มเติมด้วยครับ

*"5 หมื่น" is read as 50,000 baht; the too-short phone number is rejected by validation.*

> **Customer:** ขอโทษครับ เบอร์ที่ถูกคือ 081-234-5678
>
> **Agent:** รบกวนตรวจสอบข้อมูลครับ
> - ชื่อ: สมชาย ใจดี
> - อาชีพ: วิศวกร
> - รายได้: 50,000 บาท/เดือน
> - เบอร์ติดต่อ: 0812345678
> - ผลิตภัณฑ์ที่สนใจ: ตลอดชีพ 90/20
>
> ยืนยันว่าข้อมูลถูกต้อง และยินยอมให้ InsureX ติดต่อกลับเพื่อนำเสนอผลิตภัณฑ์ประกัน (ตาม พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล) ใช่ไหมครับ? พิมพ์ 'ยืนยัน' หรือแจ้งส่วนที่ต้องการแก้ไขได้เลยครับ

> **Customer:** ยืนยันครับ
>
> **Agent:** ขอบคุณครับคุณสมชาย ใจดี บันทึกข้อมูลเรียบร้อยแล้ว (เลขอ้างอิง LD-6A970DF4AE) ตัวแทนของ InsureX จะติดต่อกลับที่เบอร์ 0812345678 เรื่องแผน ตลอดชีพ 90/20 ครับ มีอะไรให้ช่วยเพิ่มเติมไหมครับ

*Saved to SQLite through the MCP tool (`save_lead → mcp:created`). Then the process is restarted: a new agent instance restores this session from the SQLite checkpoint (9 messages, lead ID, active product, summary).*

> **Customer:** ขอบคุณครับ อีกเรื่อง ถ้าผมเสียชีวิตตอนอายุ 60 ครอบครัวได้เงินเท่าไหร่
>
> **Agent:** ถ้าท่านเสียชีวิตตอนอายุ 60 ทายาทจะได้รับจำนวนเงินตามทุนประกันของแต่ละแผน (ผลประโยชน์กรณีเสียชีวิต) ดังนี้ครับ [1][2]
> - แผน 1 = 200,000 บาท [1]
> - แผน 2 = 300,000 บาท [1]
> - แผน 3 = 500,000 บาท [1]
> - แผน 4 = 1,000,000 บาท [1]
>
> หมายเหตุ: แผนนี้ให้ความคุ้มครองจนถึงอายุ 90 ปี ดังนั้นการเสียชีวิตที่อายุ 60 อยู่ในช่วงที่ครอบคลุมความคุ้มครองและจะได้รับผลประโยชน์ตามจำนวนเงินเอาประกันภัยที่เลือกไว้ [2]
>
> แหล่งข้อมูล:<br>
> [1] ตลอดชีพ 90/20 – ตัวอย่างผลประโยชน์ความคุ้มครอง (ไฟล์ whole_life_90_20.pdf หน้า 2)<br>
> [2] ตลอดชีพ 90/20 – เงื่อนไขการรับประกันภัย (ไฟล์ whole_life_90_20.pdf หน้า 2)

*After the restart, "this plan" still means ตลอดชีพ 90/20.*

### John (English): question → lead in three messages

> **Customer:** Hi, what is the annual premium of Whole Life 99/99 for a 40-year-old man?
>
> **Agent:** Annual premium (example) for Whole Life 99/99: 5,775 THB for a male aged 40, sum assured 300,000 THB. [1]
>
> Details:
> - Example applies to a male (age 40) with sum assured 300,000 THB; shown annual premium = 5,775 THB. [1]
> - The brochure notes premiums are fixed for the whole contract. [1]
>
> For an exact quote for your situation (different sum assured, occupation class, or other details) please contact an agent.
>
> Sources:<br>
> [1] ตลอดชีพ 99/99 – ตัวอย่างเบี้ยประกันภัยรายปี (ไฟล์ whole_life_99_99.pdf หน้า 2)

*English question, Thai brochure: the answer is in English and cites the Thai source.*

> **Customer:** What is the maximum entry age for this plan?
>
> **Agent:** Maximum entry age (age to apply) is 75 years. [1]
>
> Details:
> - Entry-age range: 1 month 1 day – 75 years (so the highest entry age is 75). [1]
> - Policy term / premium payment: coverage and premium payment run to age 99. [1]
> - Note: Applicants aged 71 and over must attach an additional rider: “ประกันภัยสุขภาพ แบบคุ้มรักษาเหมาจ่าย เอ็กตร้า”. [1]
>
> Sources:<br>
> [1] ตลอดชีพ 99/99 – เงื่อนไขการรับประกันภัย (ไฟล์ whole_life_99_99.pdf หน้า 2)

> **Customer:** I'm interested. My name is John Smith, I'm a software engineer.
>
> **Agent:** Thank you. Could you also share your monthly income, contact phone number?

> **Customer:** My income is 80,000 baht per month and my phone is +66 89 765 4321
>
> **Agent:** Please confirm your details:
> - Name: John Smith
> - Occupation: software engineer
> - Monthly income: 80,000 THB/month
> - Phone: 0897654321
> - Product of interest: ตลอดชีพ 99/99
>
> Do you confirm these details and consent to InsureX contacting you about insurance products (PDPA)? Reply 'yes' to confirm, or tell me what to change.

*"+66 89 765 4321" is normalised to 0897654321.*

> **Customer:** yes
>
> **Agent:** Thank you, John Smith! Your details are saved (reference LD-B6D7F13225). An InsureX agent will contact you at 0897654321 about ตลอดชีพ 99/99. Anything else I can help with?

The same log also shows out-of-scope questions being refused (car insurance, writing Python code), a premium the brochure doesn't contain being answered as "not in the document" instead of guessed, and both leads read back through the MCP tool `list_leads`.

## Setup

Python 3.11+ (tested on 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```

Put an API key in `.env`: `OPENAI_API_KEY=...` (default: gpt-5-mini, a few US cents for everything below).
No OpenAI key? Use a free Google Gemini key (Option B in `.env.example`), or run fully offline with no key (Option E: rule-based answers, for testing only).

```bash
python -m insurex_agent.ingestion.build_index    # read the 5 PDFs and build the search index (~20 s)
python -m pytest -q                               # 60 tests, run offline
```

## How to use

```bash
python -m insurex_agent.cli chat                  # chat with the agent
python -m insurex_agent.cli chat --debug          # ... and show the LangGraph path of every answer
```

Try: `มีประกันอะไรบ้าง` → `90/20 คืออะไร` → `ผมอายุ 35 ต้องจ่ายปีละเท่าไหร่` → `เทียบกับ 99/99 ต่างกันยังไง` → `สนใจ` → give your details → `ยืนยัน`.

In the chat: `/new` new customer · `/switch <id>` change customer · `/history` · `/leads` saved leads · `/sessions` · `/forget` erase a customer's data · `/quit`

Other commands: `python -m insurex_agent.cli leads` (list leads) · `python -m insurex_agent.cli forget <id>` · `python scripts/run_showcase.py` and `python scripts/run_demo.py` (scripted demos, written to `logs/`).

## LangGraph structure

![LangGraph](docs/graph.png)

| Part | Nodes | What happens |
|---|---|---|
| Understand | `prepare_turn`, `route` | Summarise long chats; decide what the customer wants and which product they mean |
| Answer from brochures | `retrieve`, `grade_documents`, `rewrite_query` | Hybrid search (meaning + keywords, filtered by product). If nothing useful is found, rewrite the question and search again (**cycle 1**) |
| Check the answer | `generate`, `verify_answer` | Write the answer with citations; every number must appear in the sources. If not, write it again (**cycle 2**), otherwise say "not found" (`no_answer`) |
| Collect a lead | `extract_lead`, `lead_next_step`, `save_lead`, `cancel_lead` | Take details from the customer's words, ask for what is missing, confirm + consent, save through MCP |
| Fixed replies | `catalog`, `unknown_plan`, `smalltalk`, `out_of_scope` | Product list, unknown plan codes ("99/50"), greetings, unrelated topics: no LLM needed |

Design choices:
* **The LLM never writes to the database.** Saving happens only in code, after validation and the customer's consent.
* **Deterministic guards around the LLM**: answer language, products the customer actually named, plan codes that exist, the product list.
* **Every LLM step has a rule-based fallback**, so an API outage degrades the answers instead of crashing.
* **Sessions** are LangGraph threads stored by a SQLite checkpointer; old messages are summarised so prompts stay short.

## Results

* **Answers**: 40/41 on a set of test questions with gpt-5-mini (the one miss is a correct answer worded differently from the expected text). All 4 out-of-scope questions are refused.
* **Retrieval**: with the product filter, the right brochure section is in the top 4 results for every test question.
* **Logs**: `logs/showcase_openai-gpt-5-mini.log` (a customer journey from question to saved lead, plus a second customer in parallel), `logs/demo_openai-gpt-5-mini.log` (multi-session demo with a restart), `eval/results/`.
