# Stock Analysis and Visualization

## Overview

This Streamlit app provides a platform for analyzing stock data uploaded by the user in CSV format. It generates a series of visualizations, including line plots, histograms, scatter matrices, box plots, pie charts, and candlestick charts, based on the provided data. The app also leverages Google Gemini, a Generative AI model, to analyze these visualizations and summarize the results, offering valuable insights into stock trends and market sentiment. Furthermore, it allows users to generate a downloadable PDF report based on the analysis, compiling the visualizations and AI-generated insights into a comprehensive document.

## Features

- **Upload CSV/Excel Files**: Users can upload their stock data files in various formats (CSV, XLSX, TXT).
- **Data Visualizations**: Generates multiple plots including:
  - Line plot of stock close prices.
  - Distribution plots (Distplot) for stock features (Open, High, Low, Close, Volume).
  - Box plots for stock features.
  - Histograms showing stock price trends over the years.
  - Pie chart showing the stock price movement (up/down).
  - Scatter matrix for analyzing correlations between different stock features.
  - Heatmap showing correlations above 0.9 between features.
  - Candlestick chart for detailed daily stock price movements.
- **Google Generative AI Analysis**: Analyzes images of stock visualizations and provides insights based on visual data. This is powered by Google Generative AI (Gemini 1.5 Flash).
- **Visual Analysis**: The app processes images related to stock data and generates valuable insights using Generative AI (Google Gemini).
- **Text Result Generation**: The app compiles a detailed text result summarizing the analysis.
- **PDF Report Generation**: After analysis, users can download the generated results in a PDF format.



- **PDF Report Generation**: Converts the generated insights into a downloadable PDF, formatted with proper pagination and line wrapping.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Plotly
- Google Generative AI
- ReportLab (for PDF generation)
- Pillow (for image handling)
- dotenv (for managing API keys)
- Langchain


### Setting Up the Environment

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/stock-analysis.git
    cd stock-analysis
    ```

2. Install the required Python libraries using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project directory to store your API keys:

    ```bash
    GOOGLE_API_KEY=your_google_api_key
    ```

    You can obtain your Google API key by signing up for access to [Google's Generative AI API](https://cloud.google.com/ai).

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## How It Works

### File Upload
- The app allows users to upload stock data files (`.csv`, `.xlsx`, `.txt`).
- Upon successful upload, the file is saved locally, and the system displays a message informing the user that the file is being analyzed.

### UI

UI of the app:
### 1. **Home Screen**
![User_Interface](UI&Result\User_Interface.png)

### 2. **Home Screen During File uploading**
![UI_during_file_uploading](UI&Result\UI_during_file_uploading.png)

### 3. **Home Screen After Result Generation**
![After_Result_Generation](UI&Result\After_Result_Generation.png)

### Image Analysis
- The app will look for images in a specific folder (set by `folder_path`) containing visual representations of stock data (e.g., stock price graphs, heatmaps, etc.).
- These images are analyzed by Google Generative AI, which provides insights into market stability, investor sentiment, and stock price trends based on visual data.
## Example Visualizations

Here are some example visualizations generated by the app:

### 1. **Stock Close Price Line Plot**
![Stock Close Price](graph/Line_Plot.png)

### 2. **Stock Price Distribution Plot**
![Distribution Plot](graph/Displot.png)

### 3. **Stock Price Box Plot**
![Box Plot](graph/Boxplot.png)

### 4. **Stock Price Histogram**
![Stock Price Histogram](graph/Histograms.png)

### 5. **Stock Price Pie Chart**
![Pie Chart](graph/Piechart.png)

### 6. **Stock Price Scatter Matrix**
![Scatter Matrix](graph/Scattermatrix.png)

### 7. **Stock Price Heat Map**
![Heat Map](graph/Heatmap.png)

### 8. **Candlestick Chart for Stock Movements**
![Candlestick Chart](graph/Candlestick.png)

### Text Result Generation
- The app compiles results from the image analysis and generates a text summary.
- This result is displayed in a text area on the Streamlit interface.

###  **Home Screen After Result Generation**
![After_Result_Generation](UI&Result\After_Result_Generation.png)

### PDF Report Generation
- Users can generate a PDF report containing the analysis result.
- The PDF is created dynamically with pagination to ensure that the result fits neatly within the document.
- A download button is provided for users to download the PDF.
### PDF
[Download Stock Analysis PDF Report](UI&Result\stock_analysis_report.pdf.pdf)


## Conclusion

This **Stock Analysis Application** offers a seamless, interactive, and AI-powered solution for analyzing stock data. With an intuitive interface built using **Streamlit**, users can easily upload their stock data files and generate a series of insightful visualizations. The application leverages **Google Generative AI (Gemini)** to analyze the generated charts and provide valuable insights, summarizing stock market trends, investor sentiment, and other key factors.

The app then compiles these insights into a professional **PDF report**, which can be easily downloaded and shared for further use. Whether you're an individual investor, financial analyst, or a data scientist, this app provides a comprehensive toolkit for understanding stock market data through advanced visualization and AI-based analysis.

By combining data analysis, visualization, and AI-driven insights, this application empowers users to make informed decisions based on the data at hand, without requiring deep technical expertise in data science or AI.

I hope this project helps you streamline your stock analysis and decision-making process!






