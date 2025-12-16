from transformers import pipeline
classify = "एक दीन भगवान कृष्णले भन्नु भयो, हे बालीके तीम्रो घर कहाँ हो? बालीकाले ऊत्तर दीेेेेइन मेरो घर रूस हो। "
checkpoint = "Emalper/hindi-nepali-token-classification"

classifier = pipeline("ner", model=checkpoint)
result = classifier(classify)