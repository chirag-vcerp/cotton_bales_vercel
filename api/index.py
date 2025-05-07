import os
import re
import json
import base64
import boto3
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf2image import convert_from_bytes
from openai import OpenAI
from dotenv import load_dotenv
from textractor import Textractor

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

# Set up Textractor
extractor = Textractor(profile_name="chirag16")
textract = boto3.client('textract')
# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Save directory
BASE_DIR = os.getenv("BASE_DIR", "D:\\render\\singleton_extract")

@app.route('/process-file', methods=['POST'])
def process_file():
    try:
        data = request.get_json()
        file_name = secure_filename(data.get("PDF_NAME"))
        base64_content = data.get("PDF_CONTENT")

        if not file_name or not base64_content:
            return jsonify({
                "Error_Type": "e",
                "Error_Msg": "Missing PDF_NAME or PDF_CONTENT",
                "Vehical_No": "",
                "Gross_Wt": "",
                "Tare_Wt": "",
                "Net_Wt": ""
            }), 400

        folder_name = os.path.splitext(file_name)[0]
        target_folder = os.path.join(BASE_DIR, folder_name)
        os.makedirs(target_folder, exist_ok=True)

        file_path = os.path.join(target_folder, file_name)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(base64_content))

        ext = os.path.splitext(file_name)[1].lower()
        extracted_texts = []

        if ext == ".pdf":
            images = convert_from_bytes(open(file_path, "rb").read(), dpi=300)
            for i, image in enumerate(images):
                img_path = os.path.join(target_folder, f"page_{i+1}.png")
                image.save(img_path, "PNG")
                # Use Textractor for text extraction from image
                with open(img_path, "rb") as img_file:
                    response = textract.detect_document_text(Document={'Bytes': img_file.read()})
                    extracted_texts.append(extract_text_from_textract(response))

        elif ext in [".jpg", ".jpeg", ".png"]:
            with open(file_path, "rb") as img_file:
                response = textract.detect_document_text(Document={'Bytes': img_file.read()})
                extracted_texts.append(extract_text_from_textract(response))
        else:
            return jsonify({
                "Error_Type": "e",
                "Error_Msg": "Unsupported file type",
                "Vehical_No": "",
                "Gross_Wt": "",
                "Tare_Wt": "",
                "Net_Wt": ""
            }), 400

        full_text = "\n".join(extracted_texts)
        print(repr(full_text))
        prompt = f"""
            Extract the weighbridge data from the text below and return only JSON format:
            - Do NOT hallucinate or invent any values.
            **Extraction Rules:**

            1. **Vehicle No.**: Extract the Indian vehicle registration number (vehicle number) from the following text (e.g., MH12AB1234, DL4CAF5035)
            (It will always be Indian Vehicle Number but it Can be in diff. formats e.g. Vehicle number, Truck number; remove any spaces, symbols, special characters, or newlines from it) 

            2. **Weights (Gross, Tare, Net)**:
                - Gross = Tare + Net (must validate)
                - Tare is always smallest
                - Net = Gross - Tare
            3. **If Gross values find incorrect make it correct using above logics**

            **Weight Identification Logic (Apply in Order):**

            **Rule Before Option 1**:  
            If **there are TWO standalone numbers after Vehicle No. and before 'Gross'**, **IGNORE the first** one completely — it's not a weight.

            ---

            **Option 1 (Preferred):**
            If there is **no standalone number between 'Gross' and 'Tare'**, then:
            → Take the **next 3 standalone numbers after 'Gross'** as: Gross, Tare, Net  
            → Validate with math: Gross = Tare + Net

            **Option 2:**
            If there is **only one value between Vehicle No. and 'Gross'**, then:
            - If no value between 'Gross' and 'Tare': same logic as Option 1
            - Else: weights are misplaced — look for 3 standalone numbers appearing **before their labels**

            **Option 3 (Fallback)**:
            Look for **last 3 standalone numbers in sequence** (Gross > Net > Tare), not timestamps/charges.
            - These may be up to 5 words before their label
            - Ignore dates, times, amounts like 70/-

            Text:
            \"\"\"{full_text}\"\"\" 

            Return this JSON format:
            {{
            "weigh_data": {{
                "vehicle_no": "",
                "gross_weight": "",
                "tare_weight": "",
                "net_weight": ""
            }}
            }}
            """

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.1
        )

        match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
        if match:
            try:
                parsed = json.loads(match.group())
                weigh_data = parsed.get("weigh_data", {})

                return jsonify({
                    "Error_Type": "s",
                    "Error_Msg": "",
                    "Vehical_No": weigh_data.get("vehicle_no", ""),
                    "Gross_Wt": weigh_data.get("gross_weight", ""),
                    "Tare_Wt": weigh_data.get("tare_weight", ""),
                    "Net_Wt": weigh_data.get("net_weight", "")
                })
            except json.JSONDecodeError as e:
                return jsonify({
                    "Error_Type": "e",
                    "Error_Msg": f"JSON parsing error: {str(e)}",
                    "Vehical_No": "",
                    "Gross_Wt": "",
                    "Tare_Wt": "",
                    "Net_Wt": ""
                })

        else:
            return jsonify({
                "Error_Type": "e",
                "Error_Msg": "No valid JSON found in response.",
                "Vehical_No": "",
                "Gross_Wt": "",
                "Tare_Wt": "",
                "Net_Wt": ""
            })

    except Exception as e:
        return jsonify({
            "Error_Type": "e",
            "Error_Msg": str(e),
            "Vehical_No": "",
            "Gross_Wt": "",
            "Tare_Wt": "",
            "Net_Wt": ""
        }), 500

def extract_text_from_textract(response):
    return "\n".join(
        item['Text'] for item in response.get('Blocks', [])
        if item['BlockType'] == 'LINE'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
