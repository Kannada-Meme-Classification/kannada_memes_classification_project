# kannada_memes_classification_project
Linking Memes to Knowledge Graphs for Hateful Memes classification from Kannada Datasets #TEAM Project 
👨‍🎓 Authors

Vasukinatha Adiga

Deekshith Patkar

Manish M Kumar

✨ Abstract

Memes serve as a powerful medium of digital communication, blending humor with socio-political commentary. However, they can also propagate hateful, casteist, and offensive narratives.

This project develops a multimodal pipeline to detect hateful memes in Kannada, a low-resource language, by combining:

🖼️ OCR (PyTesseract, lang=kan+eng) for text extraction

🧠 Sentiment & Transformer-based Models (mBERT, IndicBERT, sentiment-BERT)

🌐 Knowledge Graph Construction (hate lexicon + semantic mapping + Wikidata linking)

🤖 Fusion Models combining text and image embeddings

Our dataset of 140 memes was manually annotated as hateful/non-hateful. Experiments show that IndicBERT + CNN fusion achieved the best performance (71% accuracy, 0.81 F1). The knowledge graph added interpretability by linking hateful terms to semantic topics (e.g., caste, gender abuse).

📝 Introduction

Memes are widely shared in India, including Kannada-speaking regions.

They often code-mix Kannada + English, making NLP harder.

Stylized fonts challenge OCR, and cultural references make detection difficult.

Prior work focused on English memes (e.g., Hateful Memes Challenge [1]).

Our Contributions:

Curated Kannada hateful meme dataset (140 memes).

Developed a hate lexicon (~35 Kannada terms) mapped to semantic categories.

Built text, image, and fusion models for classification.

Constructed a knowledge graph for interpretability and reasoning.

📚 Related Work

Hateful Memes Challenge (Kiela et al., 2020) → introduced multimodal hate detection.

DravidianCodeMix (Chakravarthi et al., 2022) → code-mixed offensive datasets.

MemeGraphs (Kumar et al., 2023) → KG-based meme understanding.

IndicNLP Suite (Kakwani et al., 2020) → pretrained Indic language transformers.

Kannada Offensive Datasets (Hegde et al., 2024; Sharma et al., 2025) → resource creation.

⚙️ Methodology
🔡 OCR Pipeline

Tool: PyTesseract (kan+eng)

Extracted text → cleaned → stored in CSV/XLSX

OCR accuracy ~85%, manual correction for failures

🧹 Text Preprocessing

Regex cleaning, Unicode filtering, digit removal

Tokenization, stopword removal

😃 Sentiment & Text Classification

Models: IndicBERT, mBERT, sentiment-BERT

Extracted embeddings (768-dim)

Classifiers: Logistic Regression, SVM, Random Forest

🖼️ Image-only CNN

Conv2D + MaxPooling + Dense layers

Input: 128×128 resized memes

Output: Binary classification

🤝 Multimodal Fusion

CNN (image features, 512-dim) + Transformer (text, 768-dim)

Concatenated → Dense NN with Dropout

Best model: IndicBERT + CNN

🌐 Knowledge Graph (KG)

Hate lexicon (~35 Kannada words, e.g., “ರಂಡಿ”, “ಮಗನೇ”, “ಜಾತಿ”)

Mapped to topics: caste, politics, gender, vulgarity

Built with RDFLib + NetworkX

Graph structure:

Meme nodes

Keyword nodes

Topic nodes

Label nodes

📊 Dataset Description

Size: 140 memes

Source: Instagram, WhatsApp, social media

Language: Kannada + code-mixed English

Labels: Hateful (1) / Non-Hateful (0)

File: meme_extracted.xlsx

🧪 Experimental Setup

Environment: Python 3.10

Libraries: Transformers, Pandas, OpenCV, RDFLib, Matplotlib, TensorFlow

Split: Train (80%) / Test (20%)

📈 Results
Classification Metrics (Highlights)
Model	Accuracy	F1 (Macro)	Notes
Logistic Regression	54–61%	~0.58	Text-only
Random Forest	61–68%	~0.65	Text-only
SVM (mBERT)	64%	0.74	Text-only
CNN (Image)	43%	0.43	Image-only
IndicBERT + CNN	71%	0.81	Fusion
mBERT + CNN	64%	0.78	Fusion
Key Insights

OCR struggles reduced performance on stylized memes.

Fusion models outperformed single-modality models.

KG enrichment improved interpretability (caste/politics detection).

⚠️ Challenges

Small dataset (140 memes only).

OCR errors on stylized/decorative Kannada fonts.

Manual annotation was time-consuming.

Entity linking struggles for regional slang.

🚀 Future Scope

Expand dataset (>500 memes).

Fine-tune IndicBERT on Kannada hateful memes.

Apply multimodal transformers (CLIP, VisualBERT, Flamingo).

Real-time dashboard for meme moderation.

Enrich KG with Wikidata triples for broader context.

✅ Conclusion

This project demonstrates that combining OCR, IndicBERT, CNN, and Knowledge Graphs improves hateful meme detection in Kannada. While challenges remain (dataset size, OCR accuracy), our results show the potential of multimodal, culturally-aware hate detection systems in regional languages.

             

📚 References

[1] Kiela et al., “The Hateful Memes Challenge,” NeurIPS, 2020.
[2] Suryawanshi et al., “MultiOFF Dataset,” ELRA, 2020.
[3] Chakravarthi et al., “DravidianCodeMix,” LREC, 2022.
[4] Hegde et al., “Trolling Memes in Kannada,” 2024.
[5] Kakwani et al., “IndicNLP Suite,” arXiv:2005.00085, 2020.
[6] Ajawan et al., “Krishiq-BERT,” JIEI, 2024.
[7] Koutlis et al., “MemeFier,” 2023.
[8] Sharma et al., “MEMEX,” ACL, 2023.
[9] Shang et al., “KnowMeme,” IEEE, 2021.
[10] Kumar et al., “MemeGraphs,” 2023.
[11] Tommasini et al., “IMKG,” Springer, 2023.
[12] Singh et al., “Federated Hate Detection,” NAACL, 2024.
[13] Dandapat et al., “Hate Speech in Indian Languages,” Springer, 2024.
[14] Sharma et al., “HASTIKA Kannada-English Hate Corpus,” LRE, 2025.
