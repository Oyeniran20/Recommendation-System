# 🛍️ Myntra AI Product Recommender

A **hybrid content-based recommendation system** for e-commerce fashion products.  
Combines TF-IDF text similarity, structured feature matching, and Bayesian popularity scoring.

## 🚀 Live Demo
[Click here to try the app](https://recommendation-system-mhawq5iumnqbl55ifzugvg.streamlit.app/)

## 📊 Features
- **Smart Recommendations** – Find similar products based on text + specs + popularity
- **Interactive Dashboard** – Explore category distribution, rating patterns, and price trends
- **Ablation Study** – Compare hybrid vs content-only vs popularity-only modes
- **Dark Mode UI** – Modern, responsive design with custom styling

## 🧠 How It Works

| Layer | Method | Weight |
|-------|--------|--------|
| 1 | TF-IDF Text Similarity | 70% |
| 2 | Structured Features (specs, price, flags) | 30% |
| 3 | Bayesian Popularity Score | 20% |

**Final Score = 0.8 × Content Similarity + 0.2 × Popularity Score**

## 📁 Dataset
The app expects a Myntra product catalog CSV with columns:
- `product_id`, `title`, `category`, `final_price`
- `product_specifications` (JSON array)
- `amount_of_stars` (JSON object)
- `product_details` (JSON object)
- `breadcrumbs` (JSON array)

## 🛠️ Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/myntra-recommender.git
cd myntra-recommender

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
