import pandas as pd
import streamlit as st
import io
import fitz
from rapidfuzz import fuzz
import zipfile
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

import re
def main():
    
    menu = ["📁FileUploader", "🧠SmartCleanMode", "🚿FullCleanMode", "❓About"]

    choice=st.sidebar.selectbox("Clean Section",menu)
    if choice=="📁FileUploader":
        st.title("Data Cleaner-File upload section")
        uploaded_file=st.file_uploader("Choose file",type=['csv','txt','pdf','json','zip','file'])
        if uploaded_file is not None:
            st.session_state["uploaded_file"]=uploaded_file
            file_details={"filename":uploaded_file.name,"filesize":uploaded_file.size,"filetype":uploaded_file.type}
            st.write("File Details",file_details)
        try:
            if uploaded_file.name.endswith('.csv'):
                df=pd.read_csv(uploaded_file)
                st.subheader("📄Original File Content")
                st.dataframe(df.head(20))
            elif uploaded_file.name.endswith('.json'):
                df=pd.read_json(uploaded_file)
                st.subheader("📄Original File Content")
                st.dataframe(df.head(20))
            elif uploaded_file.name.endswith('.txt'):
                decode_text=uploaded_file.read().decode("utf-8")
                st.subheader("📄Original File Content")
                st.text_area("file content",decode_text,height=300)
            elif uploaded_file.name.endswith('.zip'):
                st.subheader("🗂️Original File Content")
                with zipfile.ZipFile(uploaded_file,'r') as z_f:
                    files_name=z_f.namelist()
                    st.write("Files in zip",files_name)
                    selected_file=st.selectbox("select a file",files_name)
                    with z_f.open(selected_file) as file:
                        if selected_file.endswith('.csv'):
                            df=pd.read_csv(file)
                            st.dataframe(df.head(20))
                        elif selected_file.endswith('.txt'):
                            txt_decode=file.read().decode("utf-8")
                            st.text_area(f"content of {selected_file}",txt_decode,height=300)
                        else:
                            st.warning("Only txt and csv will be previewed")
            elif uploaded_file.name.endswith('.pdf'):
                st.subheader("📄Original File Content")
                with fitz.open(stream=uploaded_file.read(),filetype="pdf") as doc:
                    page_count=doc.page_count
                    st.write(f"Total pages {page_count}")
                    page=doc.load_page(0)
                    txt=page.get_text()
                    st.text_area("PDF Content",txt,height=300)
            else:
                st.warning("⚠️file format not supported")
        except Exception as e:
            st.error("You have Not uploaded any file yet..")
    elif choice=="🧠SmartCleanMode":
        st.title("🧠Smart Clean Mode")
        def load(uploaded_file):
            df=None
            txt_data=""
            if uploaded_file.name.endswith('.csv'):
              
               df = pd.read_csv(uploaded_file)
               
            elif uploaded_file.name.endswith('.json'):
               
               df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                txt_data=uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith('.pdf'):
                with fitz.open(stream=uploaded_file.read(),filetype="pdf") as doc:
                    txt_data=""
                    for page in doc:
                        txt_data+=page.get_text()
            elif uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file,'r') as z_f:
                    files_name=z_f.namelist()
                    selected_file=st.selectbox("select a file",files_name)
                    with z_f.open(selected_file) as file:
                        if selected_file.endswith('.csv'):
                            df=pd.read_csv(file)
                            
                        elif selected_file.endswith('.txt'):
                            txt_data=file.read().decode("utf-8")
                            
                        else:
                            st.warning("Only txt and csv will be previewed")
            return df,txt_data
        uploaded_file=st.session_state.get('uploaded_file',None)
       
        if uploaded_file is None:
            st.warning("Please upload a file in 📁FileUploader section")
        else:
             #reading again from file upload section
            uploaded_file.seek(0) #already readed the file so again want to read put seek as (0)
            df,txt_data=load(uploaded_file)
           
            st.sidebar.markdown("SmartCLN_Options")
            #missing value section
            missing_value=st.sidebar.checkbox("Missing Value Handling")
            if missing_value:
                fill_option=st.sidebar.selectbox("Fill Method",["None","Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Interpolate"])
                if fill_option!="None":
                    if fill_option=="Mean":
                        df.fillna(df.mean(numeric_only=True),inplace=True)
                    elif fill_option=="Median":
                        df.fillna(df.median(numeric_only=True),inplace=True)
                    elif fill_option=="Mode":
                        df.fillna(df.mode().iloc[0],inplace=True)
                    elif fill_option=="Forward Fill":
                        df.fillna(method="ffill",inplace=True)
                    elif fill_option=="Backward Fill":
                        df.fillna(method="bfill",inplace=True)
                    elif fill_option=="Interpolate":
                        df.interpolate(inplace=True)
                drop_option=st.sidebar.multiselect("Drop Strategy", ["None","Drop Rows", "Drop Columns"])
                if drop_option:
                    if "Drop Rows" in drop_option:
                        df.dropna(axis=0,inplace=True)
                    if "Drop Columns" in drop_option:
                        df.dropna(axis=1,inplace=True)
            #duplicate
            duplicate=st.sidebar.checkbox("Duplicate Removal")
        
            if duplicate:
                dup_option= st.sidebar.radio("Duplicate Strategy", ["Remove Exact", "Fuzzy Match", "Flag Only"])
                if dup_option:
                    if dup_option=="Remove Exact":
                        df.drop_duplicates(inplace=True)
                    elif dup_option == "Fuzzy Match":
                        col_to_check = st.selectbox("Select column for fuzzy matching", df.columns)
                        threshold = st.slider("Fuzzy Match Threshold", min_value=80, max_value=100, value=90)
                        matches = []
                        used_indices = set()
                        for i in range(len(df)):
                            val_i = str(df.iloc[i][col_to_check])
                            for j in range(i + 1, len(df)):
                                val_j = str(df.iloc[j][col_to_check])
                                score = fuzz.ratio(val_i, val_j)
                                if score >= threshold:
                                    matches.append({
                                        "Index A": i,
                                        "Value A": val_i,
                                        "Index B": j,
                                        "Value B": val_j,
                                        "Similarity": score
                                        })
                                    used_indices.add(j)
                        if matches:
                            st.write("Potential Fuzzy Duplicates Found:")
                            match_df = pd.DataFrame(matches)
                            st.dataframe(match_df)
                        if st.button("✅ Confirm & Remove Fuzzy Duplicates"):
                             df.drop(index=list(used_indices), inplace=True)
                             st.success(f"🧼 Removed {len(used_indices)} fuzzy duplicate rows.")
                        else:
                            st.info("✅ No fuzzy duplicates found at this threshold.")

                    elif dup_option=="Flag Only":
                        df['is_duplicated']=df.duplicated()
                        st.info("🔍 Duplicates have been flagged in the new column `is_duplicate`.")
       
            #data type correction
            dtype=st.sidebar.checkbox("Data Type Correction")
            if dtype:
                convert_types = st.sidebar.multiselect("Convert columns to:", ["Numeric", "Date", "Categorical"])
                if convert_types:
                    if "Numeric" in convert_types:
                        numeric_columns = st.multiselect("Select columns to convert to Numeric", df.columns)
                        for col in numeric_columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    if "Date" in convert_types:
                        date_columns = st.multiselect("Select columns to convert to Date", df.columns)
                        for col in date_columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    if "Categorical" in convert_types:
                        categorical_columns = st.multiselect("Select columns to convert to Categorical", df.columns)
                        for col in categorical_columns:
                            df[col] = df[col].astype('category')
                        
            #outlier handling
            outlier=st.sidebar.checkbox("Outlier handling")
            if outlier:
                 outlier_method = st.sidebar.selectbox("Method", ["Z-Score", "IQR", "Cap/Floor"])
                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
                 columns = st.multiselect("Select columns for outlier detection", numeric_cols)

                 if columns:
                     if outlier_method == "Z-Score":
                         threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
                         for col in columns:
                             z_scores = (df[col] - df[col].mean()) / df[col].std()
                             df = df[abs(z_scores) < threshold]
                     elif outlier_method == "IQR":
                        for col in columns:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            df = df[(df[col] >= lower) & (df[col] <= upper)]
                     elif outlier_method == "Cap/Floor":
                        cap_percentile = st.slider("Cap/Floor Percentile", 0.0, 10.0, 5.0)
                        for col in columns:
                            lower = df[col].quantile(cap_percentile / 100)
                            upper = df[col].quantile(1 - cap_percentile / 100)
                            df[col] = df[col].clip(lower, upper)
                    
            #string clean
            string_clean=st.sidebar.checkbox("String Clean")
            if string_clean:
                clean_options = st.sidebar.multiselect("Choose string cleaning options",["Remove Whitespace", "Lowercase Text", "Remove Special Characters", "Fix Spelling", "Standardize Formats"])
                if df is not None:
                    text_cols = df.select_dtypes(include='object').columns.tolist()
                    selected_cols = st.multiselect("Select columns to clean", text_cols)
                    if selected_cols:
                        for col in selected_cols:
                            if "Remove Whitespace" in clean_options:
                                df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                            if "Lowercase Text" in clean_options:
                                df[col] = df[col].str.lower()
                            if "Remove Special Characters" in clean_options:
                                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)

                            if "Fix Spelling" in clean_options:
                                df[col] = df[col].apply(lambda x: str(TextBlob(str(x)).correct()) if pd.notnull(x) else x)

                            if "Standardize Formats" in clean_options:
                    
                                df[col] = df[col].str.replace(r'[^0-9]', '', regex=True)
                                df[col] = df[col].apply(lambda x: f"+91-{x[-10:]}" if len(str(x)) >= 10 else x)
                elif txt_data:
                    if "Remove Whitespace" in clean_options:
                        txt_data = re.sub(r'\s+', ' ', txt_data.strip())

                    if "Lowercase Text" in clean_options:
                        txt_data = txt_data.lower()

                    if "Remove Special Characters" in clean_options:
                        txt_data = re.sub(r'[^\w\s]', '', txt_data)

                    if "Fix Spelling" in clean_options:
                        txt_data = str(TextBlob(txt_data).correct())

                    if "Standardize Formats" in clean_options:
           
                        txt_data = re.sub(r'[^0-9]', '', txt_data)
                        if len(txt_data) >= 10:
                            txt_data = f"+91-{txt_data[-10:]}"
                    st.subheader("🧼 Cleaned Text Content")
                    st.text_area("Cleaned Output", txt_data, height=300)

            #standarization
            standard=st.sidebar.checkbox("Inconsistent data standarization")
            if standard:
                 clean_options = st.sidebar.multiselect("Method",["Normalize categories", "Standardize units", "Standard Abbreviation"])
                 normalize_category = "Normalize categories" in clean_options
                 standard_unit = "Standardize units" in clean_options
                 standard_abbreviation = "Standard Abbreviation" in clean_options
                 if df is not None and standard:
                     text_cols = df.select_dtypes(include='object').columns.tolist()
                     selected_standard_cols = st.multiselect("Select columns to standardize", text_cols)
                     if normalize_category:
                         category_map = {"m": "Male", "male": "Male", "man": "Male","f": "Female", "female": "Female", "woman": "Female","yes": "Yes", "y": "Yes","no": "No", "n": "No"}
                         for col in selected_standard_cols:
                             df[col] = df[col].astype(str).str.lower().map(category_map).fillna(df[col])
                     if standard_abbreviation:
                         abbreviation_map = {"usa": "United States","u.s.a.": "United States","us": "United States","uk": "United Kingdom","u.k.": "United Kingdom","ind": "India","in": "India"}
                         for col in selected_standard_cols:
                             df[col] = df[col].astype(str).str.lower().map(abbreviation_map).fillna(df[col])
                     if standard_unit:
                         unit_conversion_map = {"kgs": "kg", "kilograms": "kg", "kg": "kg","cms": "cm", "centimeters": "cm", "cm": "cm","meters": "m", "mtrs": "m", "m": "m"}
                         for col in selected_standard_cols:
                             df[col] = df[col].astype(str).str.lower()
                             for unit, standard in unit_conversion_map.items():
                                 df[col] = df[col].str.replace(unit, standard, regex=False)
                 elif txt_data:
                     if normalize_category:
                         txt_data = txt_data.lower()
                         category_map = {r"\b(m|male|man)\b": "Male",r"\b(f|female|woman)\b": "Female",r"\b(y|yes)\b": "Yes",r"\b(n|no)\b": "No"}
                         for pattern, replacement in category_map.items():
                             txt_data = re.sub(pattern, replacement, txt_data, flags=re.IGNORECASE)
                     if standard_abbreviation:
                         abbreviation_map = {r"\b(usa|u\.s\.a\.|us)\b": "United States",r"\b(uk|u\.k\.)\b": "United Kingdom",r"\b(ind|in)\b": "India"}
                         for pattern, replacement in abbreviation_map.items():
                             txt_data = re.sub(pattern, replacement, txt_data, flags=re.IGNORECASE)
                     if standard_unit:
                         unit_conversion_map = {r"\b(kgs|kilograms)\b": "kg",r"\b(cms|centimeters)\b": "cm",r"\b(meters|mtrs)\b": "m"}
                         for pattern, replacement in unit_conversion_map.items():
                             txt_data = re.sub(pattern, replacement, txt_data, flags=re.IGNORECASE)
                     st.subheader("🧼 Standardized Text Output")
                     st.text_area("Standardized Text", txt_data, height=300)

            #structure issue
            structure_issue=st.sidebar.checkbox("Structure Issue")
            if structure_issue:
                 structure=st.sidebar.selectbox("Options",["Columns to remove","Rename columns for clarity","Header/Footer Removal"])
                 if structure == "Columns to remove" and df is not None:
                     st.subheader("🗑️ Remove Columns")
                     remove_columns = st.multiselect("Select columns to remove", df.columns.tolist())
                     if remove_columns:
                         df.drop(columns=remove_columns, inplace=True)
                         st.success(f"✅ Removed columns: {', '.join(remove_columns)}")
                     else:
                         st.warning("⚠️ No matching columns found to remove.")
                 elif structure == "Rename columns for clarity" and df is not None:
                    st.subheader("✏️ Rename Columns")
                    rename_map = {}
                    for col in df.columns:
                        new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
                        if new_name != col:
                            rename_map[col] = new_name
                    if rename_map:
                        df.rename(columns=rename_map, inplace=True)
                        st.success("✅ Columns renamed successfully.")
                 elif structure == "Header/Footer Removal":
                    if df is not None:
                        st.subheader("📄 Remove Header/Footer Rows (CSV)")
                        num_header_rows = st.number_input("Number of top rows to remove", min_value=0, max_value=len(df), value=0)
                        num_footer_rows = st.number_input("Number of bottom rows to remove", min_value=0, max_value=len(df), value=0)
                        df = df.iloc[num_header_rows: len(df)-num_footer_rows if num_footer_rows != 0 else None]
                        st.success("✅ Header/Footer rows removed from CSV.")
                    elif txt_data:
                        st.subheader("📄 Remove Header/Footer Lines (Text)")
                        num_header_lines = st.number_input("Number of header lines to remove", min_value=0, max_value=100, value=0)
                        num_footer_lines = st.number_input("Number of footer lines to remove", min_value=0, max_value=100, value=0)
                        lines = txt_data.split('\n')
                        cleaned_lines = lines[num_header_lines: len(lines)-num_footer_lines if num_footer_lines != 0 else None]
                        txt_data = '\n'.join(cleaned_lines)
                        st.success("✅ Header/Footer lines removed from text.")
                        st.text_area("Cleaned Text Output", txt_data, height=300)
            #value normalize
            normalization=st.sidebar.checkbox("Value Normalize")
            if normalization:
                 norm_method = st.sidebar.selectbox("Normalization Method", ["Min-Max Scaling", "Z-score", "Log Transform", "Decimal Scaling"])
                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
                 norm_cols = st.multiselect("Select columns to normalize", numeric_cols)
                 if norm_cols:
                     if norm_method == "Min-Max Scaling":
                         scaler = MinMaxScaler()
                         df[norm_cols] = scaler.fit_transform(df[norm_cols])
                         st.success("✅ Applied Min-Max Scaling.")
                     elif norm_method == "Z-score":
                         scaler = StandardScaler()
                         df[norm_cols] = scaler.fit_transform(df[norm_cols])
                         st.success("✅ Applied Z-score Normalization.")
                     elif norm_method == "Log Transform":
                         for col in norm_cols:
                             df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
                             st.success("✅ Applied Log Transformation.")
                     elif norm_method == "Decimal Scaling":
                         for col in norm_cols:
                             max_val = df[col].abs().max()
                             if max_val == 0:
                                 continue
                         j = int(np.ceil(np.log10(max_val + 1)))
                         df[col] = df[col] / (10 ** j)
                         st.success("✅ Applied Decimal Scaling.")

                 
            #data validate
            validation = st.sidebar.checkbox("Data Validation")
            if validation:
                 validate=st.sidebar.selectbox("Method",["Range checks","Format validation","Consistency between fields"])
                 if validate == "Range checks" and df is not None:
                     st.subheader("📏 Range Check")
                     numeric_cols = df.select_dtypes(include='number').columns.tolist()
                     if numeric_cols:
                         col_to_check = st.selectbox("Select column for range check", numeric_cols)
                         min_val = st.number_input(f"Minimum acceptable value for {col_to_check}", value=float(df[col_to_check].min()))
                         max_val = st.number_input(f"Maximum acceptable value for {col_to_check}", value=float(df[col_to_check].max()))
                         invalid_rows = df[(df[col_to_check] < min_val) | (df[col_to_check] > max_val)]
                         if not invalid_rows.empty:
                             st.warning(f"⚠️ {len(invalid_rows)} rows found outside the range.")
                             st.dataframe(invalid_rows)
                         else:
                             st.success("✅ All values within the specified range!")
                     else:
                         st.info("ℹ️ No numeric columns found.")
                 elif validate == "Format validation" and df is not None:
                     st.subheader("format validate")
                     text_cols = df.select_dtypes(include='object').columns.tolist()
                     if text_cols:
                         col_to_check = st.selectbox("Select column for format check", text_cols)
                         format_type = st.selectbox("Select format type", ["Email", "Phone Number", "Postal Code", "Custom Regex"])
                         if format_type == "Email":
                             pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                         elif format_type == "Phone Number":
                             pattern = r'^\+?\d{10,15}$'
                         elif format_type == "Postal Code":
                             pattern = r'^\d{5,6}$'
                         else:
                             pattern = st.text_input("Enter custom regex pattern")

                         invalid_rows = df[~df[col_to_check].astype(str).str.match(pattern, na=False)]
                         if not invalid_rows.empty:
                             st.warning(f"⚠️ {len(invalid_rows)} rows failed the format validation.")
                             st.dataframe(invalid_rows)
                         else:
                             st.success("✅ All values passed the format check!")
                     else:
                         st.info("ℹ️ No text columns available for format checking.")
                 elif validate == "Consistency between fields" and df is not None:
                     st.subheader("consistency btwn fileds")
                     if len(df.columns) >= 2:
                         col1 = st.selectbox("Select first column", df.columns, key="consist_col1")
                         col2 = st.selectbox("Select second column", df.columns, key="consist_col2")
                         logic = st.selectbox("Logical condition", ["Column1 < Column2", "Column1 == Column2", "Column1 >= Column2"])
                     try:
                         if logic == "Column1 < Column2":
                             inconsistent_rows = df[df[col1] >= df[col2]]
                         elif logic == "Column1 == Column2":
                             inconsistent_rows = df[df[col1] != df[col2]]
                         elif logic == "Column1 >= Column2":
                             inconsistent_rows = df[df[col1] < df[col2]]
                         if not inconsistent_rows.empty:
                             st.warning(f"⚠️ Found {len(inconsistent_rows)} inconsistent rows.")
                             st.dataframe(inconsistent_rows)
                         else:
                             st.success("✅ All rows are logically consistent!")
                     except Exception as e:
                         st.error(f"Error during consistency check: {e}")
                 else:
                     st.info("ℹ️ Not enough columns for consistency check.")
            #noise reduction
            noise_reduction = st.sidebar.checkbox("Noise Reduction")
            if noise_reduction:
                 noise_method=st.sidebar.selectbox("Method",["Smoothing Techniques","Binning/Discretization"])
                 numeric_cols = df.select_dtypes(include='number').columns.tolist()
                 noise_cols = st.multiselect("Select columns for noise reduction", numeric_cols)
                 if noise_cols:
                     if noise_method == "Smoothing Techniques":
                         window_size = st.slider("Select moving average window size", 2, 10, 3)
                         for col in noise_cols:
                             df[col] = df[col].rolling(window=window_size, min_periods=1).mean()
                         st.success("✅ Smoothing applied using moving average.")

                     elif noise_method == "Binning/Discretization":
                         bins = st.slider("Number of bins", min_value=2, max_value=20, value=5)
                         for col in noise_cols:
                             df[col + "_binned"] = pd.cut(df[col], bins=bins, labels=False)
                         st.success("✅ Binning applied. New columns with `_binned` suffix created.")

                     
            st.subheader("🧼 Cleaned Output")
            if df is not None:
                st.dataframe(df.head(20))
            elif txt_data:
                st.text_area("Cleaned Text Data", txt_data, height=300)
            # Get the original file extension
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if df is not None:
                if file_ext == 'json':
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(label="Download cleaned JSON",data=json_data,file_name="cleaned_data.json",mime="application/json")
                else:  # default to CSV
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download cleaned CSV",data=csv_data,file_name="cleaned_data.csv",mime="text/csv")
            elif txt_data:
                text_bytes = txt_data.encode('utf-8')
                out_ext = 'txt' if file_ext in ['txt', 'pdf'] else 'txt'
                st.download_button(label="Download cleaned text",data=text_bytes,file_name=f"cleaned_data.{out_ext}",mime="text/plain")

    elif choice == "🚿FullCleanMode":
        st.title("🚿 Full Auto Clean Mode")

        def load(uploaded_file):
            df, txt_data = None, ""
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                txt_data = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith('.pdf'):
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    for page in doc:
                        txt_data += page.get_text()
            elif uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file, 'r') as z_f:
                    files_name = z_f.namelist()
                    selected_file = files_name[0]  # Auto-pick first
                    with z_f.open(selected_file) as file:
                        if selected_file.endswith('.csv'):
                            df = pd.read_csv(file)
                        elif selected_file.endswith('.txt'):
                            txt_data = file.read().decode("utf-8")
            return df, txt_data

        uploaded_file = st.session_state.get("uploaded_file", None)

        if uploaded_file is None:
            st.warning("⚠️ Please upload a file first in 📁FileUploader section.")
        else:
            uploaded_file.seek(0)
            df, txt_data = load(uploaded_file)

            if df is not None:
                st.info("📊 Detected tabular data. Performing full clean...")

                # 1. Fill missing values
                df.fillna(df.mean(numeric_only=True), inplace=True)
                st.markdown("- 🧮 Filled missing numeric values with **mean**.")

                # 2. Remove duplicates
                before = len(df)
                df.drop_duplicates(inplace=True)
                st.markdown(f"- 📛 Removed **{before - len(df)}** duplicate rows.")

               

                # 4. Normalize categories
                category_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female", "yes": "Yes", "no": "No"}
                for col in df.select_dtypes(include='object'):
                    df[col] = df[col].astype(str).str.lower().map(category_map).fillna(df[col])
                st.markdown("- 🧼 Normalized simple **categorical values**.")

                # 5. Round numerical data
                num_cols = df.select_dtypes(include='number').columns.tolist()
                df[num_cols] = df[num_cols].round(2)
                st.markdown("- 🔢 Rounded numeric columns to **2 decimals**.")

                # Show cleaned output
                st.subheader("✅ Cleaned Data Preview")
                st.dataframe(df.head(20))

            elif txt_data:
                st.info("📄 Detected plain text. Cleaning...")

                # 1. Remove whitespace
                txt_data = re.sub(r'\s+', ' ', txt_data.strip())
                st.markdown("- ✂️ Removed extra **whitespace**.")

                # 2. Lowercase
                txt_data = txt_data.lower()
                st.markdown("- 🔡 Converted to **lowercase**.")

                # 3. Remove special characters
                txt_data = re.sub(r'[^\w\s]', '', txt_data)
                st.markdown("- ❌ Removed **special characters**.")

                # Show cleaned text
                st.subheader("✅ Cleaned Text Output")
                st.text_area("Cleaned Output", txt_data, height=300)
            # Get the original file extension
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if df is not None:
                if file_ext == 'json':
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(label="Download cleaned JSON",data=json_data,file_name="cleaned_data.json",mime="application/json")
                else:  # default to CSV
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download cleaned CSV",data=csv_data,file_name="cleaned_data.csv",mime="text/csv")
            elif txt_data:
                text_bytes = txt_data.encode('utf-8')
                out_ext = 'txt' if file_ext in ['txt', 'pdf'] else 'txt'
                st.download_button(label="Download cleaned text",data=text_bytes,file_name=f"cleaned_data.{out_ext}",mime="text/plain")
    if choice== "❓About":
        
        st.title("🧹 About This App")
        st.write("Welcome to the DCleanerX – a powerful, flexible, and user-friendly tool designed to make your data cleaning process faster and easier.")
        st.header("🔍 What It Does")
        st.write("This DCleanerX helps you clean messy data files with ease, supporting multiple file formats and offering both user choose and automated cleaning options.")
        st.header("📁 Supported File Formats")
        st.markdown("""- **CSV (.csv)**
- **JSON (.json)**
- **Text Files (.txt)**
- **PDF (.pdf)** – Extracts tabular data from PDFs
- **ZIP (.zip)** – Automatically extracts and processes supported files inside""")
        st.header("⚙️ Cleaning Modes")
        st.subheader("🧠 Smart Clean Mode")
        st.write("Perfect for User's who want full control and to choose over the cleaning process")
        st.subheader("🤖 Full Clean Mode")
        st.write("Let the app do the work! Automatically detects and applies best-practice cleaning steps to your dataset")
        st.header("🚀 Why Use This App?")
        st.markdown(""" **complete the data cleaning process more faster than an human who is cleaning the dataset with their own**

**Handles multiple file types**

**Great for data analysts,data scientists,and anyone working with real-world data**""")
        st.warning("Below Files only support for some data cleaning operation So must CHECK IT..")
        st.markdown("""| Cleaning Operation                  | CSV ✅ | JSON ✅ | TXT ❌ | PDF ❌ | ZIP ✅* |
|------------------------------------|--------|----------|--------|--------|---------|
| Missing Value Handling             | ✅     | ✅       | ❌     | ❌     | ✅*     |
| Duplicate Removal                  | ✅     | ✅       | ❌     | ❌     | ✅*     |
| Data Type Correction               | ✅     | ✅       | ❌     | ❌     | ✅*     |
| Outlier Detection/Handling         | ✅     | ✅       | ❌     | ❌     | ✅*     |
| String/Text Cleaning               | ✅     | ✅       | ✅     | ✅     | ✅*     |
| Inconsistent Data Standardization  | ✅     | ✅       | ✅     | ❌     | ✅*     |
| Structural Issues                  | ✅     | ✅       | ✅     | ❌     | ✅*     |
| Value Normalization                | ✅     | ✅       | ❌     | ❌     | ✅*     |
| Data Validation                    | ✅     | ✅       | ❌     | ❌     | ✅*     |
| Noise Reduction                    | ✅     | ✅       | ❌     | ❌     | ✅*     |

*ZIP files are supported if they contain supported formats inside.
""")
                


    
if __name__=='__main__':
    main()
