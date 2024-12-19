import streamlit as st
import pandas as pd
import io
import os
import reportlab as reportlab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import glob




# Load API key from environment variables
load_dotenv()
# Congigure and Initialize the LLM with your API key
genai.configure(api_key=os.getenv("google_apikey"))
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.getenv("google_apikey"))
# Initialize the Generative AI model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Page configuration
st.set_page_config(layout='wide')

# Title
st.title("Stock Analysis")

# Subheader
st.subheader("Stock Analysis Using AI")

# Sidebar for user input
st.sidebar.subheader("Your file")

# File upload
file_uploader = st.sidebar.file_uploader(label="Upload Your CSV file.", type=['csv', 'xlsx', 'txt', 'xls'])

# Check if a file is uploaded
if file_uploader is not None:
    # Display file details
    st.write(f"Filename: {file_uploader.name}")
    st.write(f"File size: {file_uploader.size} bytes")
    uploaded_file_data = file_uploader.getvalue()

    # Define a save path
    save_dir = "file_uploaders"
    if not os.path.exists(save_dir): 
        # Create directory if it doesn't exist
        os.makedirs(save_dir) 

    save_path = os.path.join(save_dir, file_uploader.name)

    # Save the file locally
    with open(save_path, "wb") as f:
        f.write(file_uploader.getbuffer())

    st.info(f"File uploaded successfully :! Hold tight we are analysing your file.")
    

    # Folder path containing the images
    folder_path = "C:/Users/shukl/breakoutaiproject/graph/"  
    
    def analyze_images_in_folder(folder_path):
        """
                Provide a high-level assessment of the company's market stability and investor sentiment based on the visual data.
        Analyzes all images in the specified folder and compiles the results.
        Return a concise yet detailed paragraph summarizing these insights in a way that's suitable for an executive report. If possible, quantify trends using numerical data observed in the graphs."
        
        Args:
            folder_path (str): Path to the folder containing images.
        
        Returns:
            str: Compiled results of the analysis for all images in a paragraph format giving the best analysis result.
        """
        # Get all image files in the folder (supports common image formats)
        image_paths = glob.glob(os.path.join(folder_path, '*.png')) + \
                  glob.glob(os.path.join(folder_path, '*.jpg')) + \
                  glob.glob(os.path.join(folder_path, '*.jpeg'))
    
        compiled_results = []  # List to store results
    
        for image_path in image_paths:
            try:
                # Open the image
                photo = Image.open(image_path)
                
                # Analyze the image using Generative AI
                response = model.generate_content(["Analyze the image and give valuable insights", photo])
                
                # Append result to compiled_results
                compiled_results.append(f"Result for {image_path}:\n{response}\n{'-' * 50}")
            
            except Exception as e:
                # Handle errors during image processing or API calls
                compiled_results.append(f"Error processing {image_path}: {e}\n{'-' * 50}")
        
        # Join all results into a single string and return
        return "\n".join(compiled_results)

    # Analyze images in the folder and compile the response
    compiled_response = analyze_images_in_folder(folder_path)


    st.info(f"Analysis Completed. Please Click Generate")
    # Button to trigger analysis
    if st.button("Generate"):
        with st.spinner("Please hold on while the code is generated..."):
            try:
                    compiled_response = llm.invoke(compiled_response)
                    # Call compiled_response with the uploaded file path
                    result_content = compiled_response

                    # Display the result in a text area
                    st.text_area("Generated Result", result_content, height=300)

                    # Function to generate PDF in memory with pagination
                    def generate_pdf(content, right_margin=100):
                        # Create a bytes buffer to store the PDF
                        buffer = io.BytesIO()

                        # Create a canvas for the PDF
                        c = canvas.Canvas(buffer, pagesize=letter)
                        width, height = letter  # Dimensions of the page

                        # Set up the PDF font
                        c.setFont("Helvetica", 12)

                        # Title for the PDF
                        c.drawString(100, height - 50, "Stock Analysis Report")
                        c.drawString(100, height - 70, "----------------------------------")

                        # Define the content area and line spacing
                        x_position = 100  # Left margin
                        y_position = height - 100  # Top margin
                        line_spacing = 14  # Space between lines

                        # Right position for the margin (page width minus right margin)
                        right_position = width - right_margin
                        left_position = x_position

                        # Loop through the content and handle pagination
                        for line in content.splitlines():
                            # Calculate the width of the line and check if it will fit
                            line_width = c.stringWidth(line, "Helvetica", 12)
                            if line_width > (right_position - left_position):
                                # If the line is too long, break it into multiple lines (simple word wrap)
                                words = line.split()
                                wrapped_line = ""
                                for word in words:
                                    # Check if adding this word will exceed the page width
                                    if c.stringWidth(wrapped_line + word, "Helvetica", 12) < (right_position - left_position):
                                        wrapped_line += " " + word
                                    else:
                                        # Draw the wrapped line and start a new line
                                        c.drawString(left_position, y_position, wrapped_line.strip())
                                        y_position -= line_spacing
                                        wrapped_line = word

                                # Draw the remaining wrapped line
                                c.drawString(left_position, y_position, wrapped_line.strip())
                            else:
                                # If the line fits within the allowed width, draw it
                                c.drawString(left_position, y_position, line)
                            
                            y_position -= line_spacing  # Move down for the next line

                            # If the text gets too close to the bottom, create a new page
                            if y_position < 100:  # Bottom margin
                                c.showPage()
                                c.setFont("Helvetica", 12)
                                y_position = height - 100  # Reset the position for the new page

                        # Finish the PDF
                        c.showPage()  # Ensure the last page is properly saved
                        c.save()

                        # Move buffer position to the beginning
                        buffer.seek(0)

                        return buffer

                    # Generate PDF from the result content
                    pdf_buffer = generate_pdf(result_content)

                    # Add a download button to download the PDF
                    st.download_button(
                        label="Download PDF",
                        data=pdf_buffer,
                        file_name="result.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")
