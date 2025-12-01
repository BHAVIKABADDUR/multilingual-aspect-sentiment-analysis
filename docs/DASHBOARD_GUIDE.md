# 📊 Dashboard User Guide
**E-commerce Sentiment Analysis Dashboard**  
**By: Bhavika Baddur**

---

## 🚀 **Quick Start**

### **Run the Dashboard:**
```bash
streamlit run app.py
```

The dashboard will open automatically in your web browser at `http://localhost:8501`

---

## 📑 **Dashboard Pages**

### **1. 📊 Overview**
- Overall sentiment statistics (Positive, Negative, Neutral %)
- Sentiment distribution pie chart
- Rating distribution (1-5 stars)
- Language and category breakdowns
- Key metrics at a glance

### **2. 🎯 Aspect Analysis**
- Performance summary for all 5 aspects
- Interactive sentiment heatmap
- Critical insights and recommendations
- Identifies strengths and weaknesses

**5 Business Aspects:**
- Product Quality
- Delivery
- Packaging
- Price
- Customer Service

### **3. 🏪 Platform Insights**
- Sentiment distribution by platform
- Platform comparison
- Average ratings per platform
- Review volume statistics

**Platforms:** Amazon, Flipkart, Myntra, Nykaa, Swiggy, Zomato

### **4. 💬 Review Analyzer**
- Real-time sentiment prediction
- Test custom reviews
- See confidence scores
- Try sample reviews
- Supports English, Hindi, and Hinglish

### **5. 📈 About**
- Project overview
- Key achievements
- Technology stack
- Business value & ROI
- Contact information

---

## 🎨 **Features**

### **Interactive Visualizations:**
- ✅ Pie charts for sentiment distribution
- ✅ Bar charts for ratings and categories
- ✅ Heatmaps for aspect sentiments
- ✅ Stacked bars for platform comparison

### **Real-Time Analysis:**
- ✅ Type any review and get instant sentiment
- ✅ See confidence scores
- ✅ Works with multilingual text

### **Business Insights:**
- ✅ Color-coded performance indicators
- ✅ Critical issue highlights
- ✅ Actionable recommendations
- ✅ ROI projections

---

## 🎯 **How to Use**

### **For Stakeholders:**
1. Start with **📊 Overview** to see overall performance
2. Check **🎯 Aspect Analysis** for detailed insights
3. Review critical issues and recommendations
4. Use insights for strategic decisions

### **For Developers:**
1. Explore **🏪 Platform Insights** for data patterns
2. Test **💬 Review Analyzer** with custom reviews
3. Check **📈 About** for technical details

### **For Demos:**
1. Show **📊 Overview** first (impressive visuals)
2. Highlight **🎯 Aspect Analysis** (business value)
3. Demo **💬 Review Analyzer** (interactive feature)

---

## 💡 **Tips**

### **Navigation:**
- Use the sidebar to switch between pages
- Metrics are automatically calculated
- All charts are interactive (hover for details)

### **Custom Analysis:**
- Enter any review in the Review Analyzer
- Mix English and Hindi words
- Get instant sentiment prediction

### **Performance:**
- Dashboard loads 20,000 reviews efficiently
- Data is cached for fast performance
- Refresh browser to reload data

---

## 🔧 **Troubleshooting**

### **Dashboard won't start?**
```bash
# Install dependencies
pip install streamlit plotly

# Run dashboard
streamlit run app.py
```

### **Data not loading?**
- Ensure `processed_data/dataset_with_aspects.csv` exists
- Run `python scripts/05_aspect_extraction.py` first

### **Models not found?**
- Ensure `models/sentiment_model.pkl` exists
- Run `python scripts/04_simple_sentiment_analysis.py` first

### **Port already in use?**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## 📊 **Dashboard Controls**

### **Sidebar Info:**
- Total reviews count
- Model accuracy (76.71%)
- Aspects analyzed (5)

### **Interactive Elements:**
- Click on charts to zoom
- Hover for detailed tooltips
- Use buttons to try samples

---

## 🎓 **Key Insights You'll See**

### **Strengths (Green):**
- ✅ Product Quality: 69.9% positive
- ✅ Price: 61.7% positive

### **Critical Issues (Red):**
- ❌ Customer Service: 36.8% positive
- ❌ Packaging: 39.3% positive
- ⚠️ Delivery: 45.1% positive

### **ROI Projection:**
- Investment: ₹10-16 Lakhs
- Return: ₹33-45 Lakhs/year
- Payback: 3-6 months
- Net ROI: 200-300%

---

## 🚀 **Next Steps**

After exploring the dashboard:

1. **Share with stakeholders** - Professional presentation-ready
2. **Use for decisions** - Data-driven insights
3. **Export findings** - Take screenshots or notes
4. **Implement recommendations** - Follow action items

---

## 📞 **Support**

For questions or issues:
- Check `docs/` folder for more documentation
- Review `EXECUTIVE_SUMMARY.md` for business insights
- See `PROJECT_STATUS.md` for technical details

---

**Created by:** Bhavika Baddur  
**Project:** Multilingual Customer Intelligence Platform  
**Date:** November 2025  
**Version:** 1.0
