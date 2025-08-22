# Grocery Recommendation System with Market Basket Analysis
# Features: Content-based, Association Rules, Explainable AI, Streamlit Dashboard
# Using Real Instacart Dataset

import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings('ignore')

class GroceryRecommendationSystem:
    def __init__(self):
        self.products_df = None
        self.transactions_df = None
        self.orders_df = None
        self.order_products_df = None
        self.association_rules_df = None
        self.content_similarity_matrix = None
        self.vectorizer = None
        
    def load_instacart_data(self, base_path=None):
        """Load real Instacart dataset"""
        
        if base_path is None:
            base_path = r"C:\Users\dellg\Desktop\rec_system"
        
        try:
            # Load the datasets
            print("📊 Loading Instacart datasets...")
            
            # Load products data
            products_path = os.path.join(base_path, "products.csv")
            self.products_df = pd.read_csv(products_path)
            print(f"✅ Products loaded: {len(self.products_df)} products")
            
            # Load orders data
            orders_path = os.path.join(base_path, "orders.csv")
            self.orders_df = pd.read_csv(orders_path)
            print(f"✅ Orders loaded: {len(self.orders_df)} orders")
            
            # Load order products data
            order_products_path = os.path.join(base_path, "order_products__train.csv")
            self.order_products_df = pd.read_csv(order_products_path)
            print(f"✅ Order products loaded: {len(self.order_products_df)} order items")
            
            # Merge data to create transactions
            self.create_transactions_data()
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print("📁 Make sure these files exist in your folder:")
            print("  - products.csv")
            print("  - orders.csv") 
            print("  - order_products__train.csv")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_transactions_data(self):
        """Create transaction data by merging orders and products"""
        
        print("🔄 Creating transaction data...")
        
        # Sample data for faster processing (use first 25,000 orders for demo)
        sample_orders = self.orders_df.head(25000)  # Reduced from 50k
        
        # Merge order_products with sampled orders
        transactions = self.order_products_df[
            self.order_products_df['order_id'].isin(sample_orders['order_id'])
        ]
        
        # Merge with products to get product names
        transactions = transactions.merge(
            self.products_df[['product_id', 'product_name', 'aisle_id', 'department_id']], 
            on='product_id', 
            how='left'
        )
        
        # Clean product names and handle missing values
        transactions = transactions.dropna(subset=['product_name'])
        transactions['product_name'] = transactions['product_name'].str.title()
        
        # Create final transactions dataframe
        self.transactions_df = transactions[['order_id', 'product_id', 'product_name', 'add_to_cart_order']].copy()
        self.transactions_df.rename(columns={'order_id': 'transaction_id', 'add_to_cart_order': 'quantity'}, inplace=True)
        
        # Clean and enhance products dataframe - only keep products that appear in transactions
        transaction_products = set(self.transactions_df['product_name'].unique())
        self.products_df = self.products_df[self.products_df['product_name'].str.title().isin(transaction_products)].copy()
        self.products_df['product_name'] = self.products_df['product_name'].str.title()
        
        # Create better category mapping using department_id
        dept_mapping = {
            1: 'Frozen', 2: 'Other', 3: 'Bakery', 4: 'Produce', 5: 'Alcohol', 
            6: 'International', 7: 'Beverages', 8: 'Pets', 9: 'Dry Goods Pasta',
            10: 'Bulk', 11: 'Personal Care', 12: 'Meat Seafood', 13: 'Pantry',
            14: 'Breakfast', 15: 'Canned Goods', 16: 'Dairy Eggs', 17: 'Household',
            18: 'Babies', 19: 'Snacks', 20: 'Deli', 21: 'Missing'
        }
        
        self.products_df['category'] = self.products_df['department_id'].map(dept_mapping).fillna('Other')
        
        # Create descriptions
        self.products_df['description'] = (
            "Quality " + self.products_df['product_name'].str.lower() + 
            " from " + self.products_df['category'].str.lower() + " section"
        )
        
        print(f"✅ Transaction data created: {len(self.transactions_df):,} items in {len(self.transactions_df['transaction_id'].unique()):,} transactions")
        print(f"✅ Products filtered to {len(self.products_df):,} items that appear in transactions")
        
    def generate_synthetic_data(self, n_products=200, n_transactions=10000):
        """Generate realistic synthetic grocery data (fallback if real data fails)"""
        
        print("📊 Generating synthetic data as fallback...")
        
        # Product categories and items
        categories = {
            'Fruits': ['Apple', 'Banana', 'Orange', 'Grapes', 'Strawberry', 'Mango', 'Pineapple', 'Watermelon'],
            'Vegetables': ['Carrot', 'Broccoli', 'Spinach', 'Tomato', 'Onion', 'Potato', 'Bell Pepper', 'Cucumber'],
            'Dairy': ['Milk', 'Cheese', 'Yogurt', 'Butter', 'Cream', 'Ice Cream'],
            'Meat': ['Chicken', 'Beef', 'Pork', 'Fish', 'Turkey', 'Salmon'],
            'Bakery': ['Bread', 'Bagels', 'Croissant', 'Muffins', 'Cake', 'Cookies'],
            'Beverages': ['Coffee', 'Tea', 'Juice', 'Soda', 'Water', 'Energy Drink'],
            'Snacks': ['Chips', 'Crackers', 'Nuts', 'Chocolate', 'Candy', 'Popcorn'],
            'Pantry': ['Rice', 'Pasta', 'Cereal', 'Oil', 'Salt', 'Sugar', 'Flour'],
            'Frozen': ['Frozen Pizza', 'Ice Cream', 'Frozen Vegetables', 'Frozen Fruits'],
            'Personal Care': ['Shampoo', 'Soap', 'Toothpaste', 'Deodorant']
        }
        
        # Generate products
        products = []
        product_id = 1
        
        for category, items in categories.items():
            for item in items:
                products.append({
                    'product_id': product_id,
                    'product_name': item,
                    'category': category,
                    'price': np.random.uniform(1, 20),
                    'description': f"Fresh {item.lower()} from {category.lower()}"
                })
                product_id += 1
        
        self.products_df = pd.DataFrame(products)
        
        # Generate transactions with realistic patterns
        transactions = []
        
        # Define common item combinations (market basket patterns)
        common_combos = [
            ['Bread', 'Butter', 'Milk'],
            ['Coffee', 'Sugar', 'Cream'],
            ['Pasta', 'Tomato', 'Cheese'],
            ['Chicken', 'Rice', 'Oil'],
            ['Apple', 'Banana', 'Orange'],
            ['Chips', 'Soda', 'Candy'],
            ['Shampoo', 'Soap', 'Toothpaste'],
            ['Ice Cream', 'Chocolate', 'Cookies'],
            ['Fish', 'Rice', 'Oil'],
            ['Cereal', 'Milk', 'Banana']
        ]
        
        for transaction_id in range(1, n_transactions + 1):
            # Randomly choose between combo shopping and individual items
            if np.random.random() < 0.4:  # 40% combo shopping
                combo = np.random.choice(len(common_combos))
                items = common_combos[combo].copy()
                # Add some random items
                additional_items = np.random.choice(self.products_df['product_name'].values, 
                                                 size=np.random.randint(0, 3), replace=False)
                items.extend(additional_items)
            else:
                # Random shopping
                n_items = np.random.randint(1, 8)
                items = np.random.choice(self.products_df['product_name'].values, 
                                       size=n_items, replace=False)
            
            for item in items:
                product_id = self.products_df[self.products_df['product_name'] == item]['product_id'].iloc[0]
                transactions.append({
                    'transaction_id': transaction_id,
                    'product_id': product_id,
                    'product_name': item,
                    'quantity': np.random.randint(1, 4)
                })
        
        self.transactions_df = pd.DataFrame(transactions)
        
    def build_content_based_recommender(self):
        """Build content-based recommender using product features"""
        
        print("🔍 Building content-based recommender...")
        
        # For large datasets, we'll use a subset of products for similarity matrix
        # but keep all products for recommendations
        max_products_for_similarity = 5000  # Manageable size
        
        if len(self.products_df) > max_products_for_similarity:
            print(f"📊 Dataset too large ({len(self.products_df)} products). Using top {max_products_for_similarity} most popular products for similarity calculation...")
            
            # Get most popular products from transactions
            popular_products = self.transactions_df['product_name'].value_counts().head(max_products_for_similarity)
            similarity_products = self.products_df[self.products_df['product_name'].isin(popular_products.index)].copy()
            
            print(f"✅ Using {len(similarity_products)} most popular products for content similarity")
        else:
            similarity_products = self.products_df.copy()
        
        # Create feature vectors for similarity calculation
        features = similarity_products['category'] + ' ' + similarity_products['description']
        
        # TF-IDF Vectorization with reduced features
        self.vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        feature_matrix = self.vectorizer.fit_transform(features)
        
        # Compute similarity matrix for subset
        self.content_similarity_matrix = cosine_similarity(feature_matrix)
        self.similarity_products_df = similarity_products.reset_index(drop=True)
        
        print("✅ Content-based recommender built with optimized memory usage!")
        
    def build_market_basket_analysis(self, min_support=0.005, min_threshold=0.1):
        """Build association rules using Market Basket Analysis"""
        
        print("🛍️ Building market basket analysis...")
        
        # Create basket matrix (transaction x product)
        basket = self.transactions_df.groupby(['transaction_id', 'product_name'])['quantity'].sum().unstack().fillna(0)
        
        # Convert to boolean (bought or not)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        print(f"📊 Basket matrix created: {basket_sets.shape[0]} transactions x {basket_sets.shape[1]} products")
        
        # Find frequent itemsets with lower support for real data
        try:
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True, verbose=1)
            print(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
            
            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
                
                # Clean up the rules
                rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0] if len(x) == 1 else list(x))
                rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0] if len(x) == 1 else list(x))
                
                self.association_rules_df = rules.sort_values('lift', ascending=False)
                print(f"✅ Generated {len(rules)} association rules")
            else:
                print("⚠️ No frequent itemsets found, trying lower support...")
                frequent_itemsets = apriori(basket_sets, min_support=0.001, use_colnames=True)
                if len(frequent_itemsets) > 0:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.05)
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0] if len(x) == 1 else list(x))
                    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0] if len(x) == 1 else list(x))
                    self.association_rules_df = rules.sort_values('lift', ascending=False)
                else:
                    self.association_rules_df = pd.DataFrame()
                    print("❌ No association rules found")
        except Exception as e:
            print(f"❌ Error in market basket analysis: {e}")
            self.association_rules_df = pd.DataFrame()
    
    def get_content_recommendations(self, product_name, n_recommendations=5):
        """Get content-based recommendations"""
        
        # Check if product exists in similarity matrix
        if not hasattr(self, 'similarity_products_df'):
            # Fallback to category-based recommendations
            return self.get_category_based_recommendations(product_name, n_recommendations)
        
        if product_name not in self.similarity_products_df['product_name'].values:
            # If product not in similarity matrix, use category-based fallback
            return self.get_category_based_recommendations(product_name, n_recommendations)
        
        # Get product index in similarity matrix
        product_idx = self.similarity_products_df[self.similarity_products_df['product_name'] == product_name].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity_matrix[product_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar products (excluding the input product)
        recommendations = []
        for idx, score in sim_scores[1:n_recommendations+1]:
            product_info = self.similarity_products_df.iloc[idx]
            recommendations.append({
                'product_name': product_info['product_name'],
                'category': product_info['category'],
                'similarity_score': round(score, 3),
                'reason': f"Similar category and features (similarity: {round(score, 3)})"
            })
        
        return recommendations, "Content-based recommendations"
    
    def get_category_based_recommendations(self, product_name, n_recommendations=5):
        """Fallback category-based recommendations"""
        
        if product_name not in self.products_df['product_name'].values:
            return [], "Product not found"
        
        # Get product category
        product_category = self.products_df[self.products_df['product_name'] == product_name]['category'].iloc[0]
        
        # Get other products in same category
        same_category_products = self.products_df[
            (self.products_df['category'] == product_category) & 
            (self.products_df['product_name'] != product_name)
        ]
        
        # If we have transaction data, prioritize popular products
        if hasattr(self, 'transactions_df') and len(self.transactions_df) > 0:
            # Get popularity scores
            popularity = self.transactions_df['product_name'].value_counts()
            same_category_products = same_category_products.copy()
            same_category_products['popularity'] = same_category_products['product_name'].map(popularity).fillna(0)
            same_category_products = same_category_products.sort_values('popularity', ascending=False)
        
        # Get top recommendations
        recommendations = []
        for _, product_info in same_category_products.head(n_recommendations).iterrows():
            recommendations.append({
                'product_name': product_info['product_name'],
                'category': product_info['category'],
                'similarity_score': 0.5,  # Default similarity for same category
                'reason': f"Same category ({product_category}) - popular item"
            })
        
        return recommendations, "Category-based recommendations"
    
    def get_association_recommendations(self, product_name, n_recommendations=5):
        """Get association rule-based recommendations"""
        
        if self.association_rules_df.empty:
            return [], "No association rules found"
        
        # Find rules where the product is an antecedent
        relevant_rules = self.association_rules_df[
            self.association_rules_df['antecedents'].apply(
                lambda x: product_name in x if isinstance(x, list) else x == product_name
            )
        ]
        
        recommendations = []
        for _, rule in relevant_rules.head(n_recommendations).iterrows():
            consequent = rule['consequents'] if isinstance(rule['consequents'], str) else rule['consequents'][0]
            recommendations.append({
                'product_name': consequent,
                'confidence': round(rule['confidence'], 3),
                'lift': round(rule['lift'], 3),
                'support': round(rule['support'], 3),
                'reason': f"Frequently bought together (confidence: {round(rule['confidence']*100, 1)}%)"
            })
        
        return recommendations, "Market Basket Analysis recommendations"
    
    def get_hybrid_recommendations(self, product_name, n_recommendations=10):
        """Get hybrid recommendations (content + association)"""
        
        content_recs, content_msg = self.get_content_recommendations(product_name, n_recommendations//2)
        assoc_recs, assoc_msg = self.get_association_recommendations(product_name, n_recommendations//2)
        
        return {
            'content_based': {
                'recommendations': content_recs,
                'message': content_msg
            },
            'association_based': {
                'recommendations': assoc_recs,
                'message': assoc_msg
            }
        }

def main():
    st.set_page_config(page_title="Grocery Recommender", page_icon="🛒", layout="wide")
    
    st.title("🛒 Smart Grocery Recommendation System")
    st.markdown("### AI-Powered Recommendations with Real Instacart Data")
    
    # Initialize the system
    if 'recommender' not in st.session_state:
        with st.spinner("🔄 Initializing recommendation system with real data..."):
            st.session_state.recommender = GroceryRecommendationSystem()
            
            # Try to load real data first
            if st.session_state.recommender.load_instacart_data():
                st.success("✅ Real Instacart dataset loaded successfully!")
                st.info("📊 Using 25,000 real transactions with memory optimization for optimal performance")
            else:
                st.warning("⚠️ Could not load real data, using synthetic data")
                st.session_state.recommender.generate_synthetic_data()
            
            st.session_state.recommender.build_content_based_recommender()
            st.session_state.recommender.build_market_basket_analysis()
    
    recommender = st.session_state.recommender
    
    # Sidebar
    st.sidebar.header("🎯 System Features")
    st.sidebar.markdown("""
    - **Real Dataset**: Instacart grocery orders
    - **Content-Based Filtering**: Similar products by features
    - **Market Basket Analysis**: Frequently bought together
    - **Hybrid Approach**: Best of both worlds
    - **Explainable AI**: Clear reasoning for recommendations
    """)
    
    # Dataset info
    st.sidebar.header("📊 Dataset Info")
    if recommender.products_df is not None:
        st.sidebar.write(f"**Products:** {len(recommender.products_df):,}")
        st.sidebar.write(f"**Transactions:** {len(recommender.transactions_df['transaction_id'].unique()):,}")
        st.sidebar.write(f"**Total Items:** {len(recommender.transactions_df):,}")
        
        if not recommender.association_rules_df.empty:
            st.sidebar.write(f"**Association Rules:** {len(recommender.association_rules_df):,}")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Select Product")
        
        # Product selection
        if recommender.products_df is not None:
            categories = sorted(recommender.products_df['category'].unique())
            selected_category = st.selectbox("Choose Category", ['All'] + list(categories))
            
            if selected_category == 'All':
                available_products = sorted(recommender.products_df['product_name'].tolist())
            else:
                available_products = sorted(recommender.products_df[
                    recommender.products_df['category'] == selected_category
                ]['product_name'].tolist())
            
            selected_product = st.selectbox("Choose Product", available_products)
            
            # Recommendation settings
            st.subheader("⚙️ Settings")
            n_recommendations = st.slider("Number of recommendations", 1, 10, 5)
            
            if st.button("🎯 Get Recommendations", type="primary"):
                st.session_state.show_recommendations = True
    
    with col2:
        st.header("🎯 Recommendations")
        
        if 'show_recommendations' in st.session_state and st.session_state.show_recommendations and recommender.products_df is not None:
            
            # Get hybrid recommendations
            hybrid_results = recommender.get_hybrid_recommendations(selected_product, n_recommendations)
            
            # Display selected product info
            product_info = recommender.products_df[
                recommender.products_df['product_name'] == selected_product
            ].iloc[0]
            
            st.info(f"**Selected Product:** {selected_product} ({product_info['category']})")
            
            # Content-based recommendations
            st.subheader("🔍 Similar Products (Content-Based)")
            content_recs = hybrid_results['content_based']['recommendations']
            
            if content_recs:
                for i, rec in enumerate(content_recs, 1):
                    with st.expander(f"{i}. {rec['product_name']} ({rec['category']})"):
                        st.write(f"**Reason:** {rec['reason']}")
                        st.write(f"**Similarity Score:** {rec['similarity_score']}")
            else:
                st.write("No content-based recommendations found.")
            
            # Association-based recommendations
            st.subheader("🛍️ Frequently Bought Together (Market Basket)")
            assoc_recs = hybrid_results['association_based']['recommendations']
            
            if assoc_recs:
                for i, rec in enumerate(assoc_recs, 1):
                    with st.expander(f"{i}. {rec['product_name']}"):
                        st.write(f"**Reason:** {rec['reason']}")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Confidence", f"{rec['confidence']}")
                        col_b.metric("Lift", f"{rec['lift']}")
                        col_c.metric("Support", f"{rec['support']}")
            else:
                st.write("No association-based recommendations found.")
                st.info("💡 Try products like 'Banana', 'Milk', or 'Bread' which are more common in transactions")
    
    # Analytics Dashboard
    if recommender.products_df is not None:
        st.header("📊 System Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Top Products", "Category Distribution"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Products", f"{len(recommender.products_df):,}")
            col2.metric("Total Transactions", f"{len(recommender.transactions_df['transaction_id'].unique()):,}")
            col3.metric("Categories", len(recommender.products_df['category'].unique()))
            
            st.subheader("Sample Transaction Data")
            st.dataframe(recommender.transactions_df.head(10))
            
            st.subheader("Sample Products Data")
            st.dataframe(recommender.products_df[['product_name', 'category', 'description']].head(10))
        
        with tab2:
            # Most popular products
            popular_products = recommender.transactions_df['product_name'].value_counts().head(15)
            fig = px.bar(x=popular_products.values, y=popular_products.index, orientation='h',
                         title="Top 15 Most Popular Products",
                         labels={'x': 'Number of Orders', 'y': 'Product'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Category distribution
            category_counts = recommender.products_df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index,
                         title="Product Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
            
            # Category popularity in transactions
            category_transactions = recommender.transactions_df.merge(
                recommender.products_df[['product_name', 'category']], 
                on='product_name'
            )['category'].value_counts()
            
            fig2 = px.bar(x=category_transactions.index, y=category_transactions.values,
                         title="Category Popularity in Transactions",
                         labels={'x': 'Category', 'y': 'Number of Items Ordered'})
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()