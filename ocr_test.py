import easyocr

reader = easyocr.Reader(['en'])
results = reader.readtext("ocr_test.jpg", detail=1, paragraph=False)

if not results:
    print("No text detected.")
else:
    for (bbox, text, confidence) in results:
        print(f"Text: {text} | Confidence: {confidence:.2f}")