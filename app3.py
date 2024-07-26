import streamlit as st
from streamlit_extras.colored_header import colored_header
from datetime import datetime
import logging
import os
import re
import io
import zipfile
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import yaml
import pyotp
import qrcode
import streamlit_authenticator as stauth
import pandas as pd
import pyodbc
from langchain_community.chat_models import AzureChatOpenAI
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
from yaml.loader import SafeLoader

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="QA on Agent Calls", page_icon="💬", layout="centered")

# Load environment variables
load_dotenv()
connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
container_name = "qacallagent"
blob_name = "config/config.yaml"
SHAREPOINT_LINK = os.getenv("SHAREPOINT_LINK")

# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def load_config():
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    return yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

def save_config(config):
    blob_client = container_client.get_blob_client(blob_name)
    updated_blob_data = yaml.dump(config)
    blob_client.upload_blob(updated_blob_data, overwrite=True)

config = load_config()

# Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# Initialize session state variables for authentication and OTP
if "name" not in st.session_state:
    st.session_state.name = None

if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

if "username" not in st.session_state:
    st.session_state.username = None

if 'otp_setup_complete' not in st.session_state:
    st.session_state['otp_setup_complete'] = False

if 'otp_verified' not in st.session_state:
    st.session_state['otp_verified'] = False

if 'show_qr_code' not in st.session_state:
    st.session_state['show_qr_code'] = False

def setup_2fa(username, config):
    user_data = config['credentials']['usernames'][username]
    otp_secret = user_data.get('otp_secret', "")

    if not otp_secret:
        otp_secret = pyotp.random_base32()
        config['credentials']['usernames'][username]['otp_secret'] = otp_secret
        save_config(config)
        st.session_state['otp_setup_complete'] = False
        st.session_state['show_qr_code'] = True
        logger.info("Generated new OTP secret and set show_qr_code to True")
    else:
        st.session_state['otp_setup_complete'] = True

    totp = pyotp.TOTP(otp_secret)

    if not st.session_state['otp_verified']:
        if st.session_state['show_qr_code']:
            logger.info("Displaying QR code for initial OTP setup")
            otp_uri = totp.provisioning_uri(name=user_data['email'], issuer_name="QA on Agent Calls")
            qr = qrcode.make(otp_uri)
            qr = qr.resize((200, 200))  # Resize the QR 

            st.image(qr, caption="Scan this QR code with your authenticator app")

        otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
        verify_button_clicked = st.button("Verify OTP")

        if verify_button_clicked:
            if totp.verify(otp_input):
                st.session_state['otp_verified'] = True
                st.session_state['show_qr_code'] = False
                save_config(config)
                st.experimental_rerun()
            else:
                st.error("Invalid OTP. Please try again.")
    else:
        return True
    return False

# Authentication
name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    if setup_2fa(username, config):
        sharepoint_link = os.getenv("SHAREPOINT_LINK")

        # Constants
        AUDIO_FILE_EXTENSION = '.wav'
        local_folder = "./qa_call_analysis_audio_files"
        output_folder = "./qa_call_analysis_output_files"

        # Azure Blob Storage configuration
        azure_blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        azure_blob_container_name = "container-qa-analysis"

        # Azure OpenAI configuration
        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_ENDPOINT")

        # MSSQL database configuration
        mssql_server = os.getenv("MSSQL_SERVER")
        mssql_database = os.getenv("MSSQL_DATABASE")
        mssql_username = os.getenv("MSSQL_USERNAME")
        mssql_password = os.getenv("MSSQL_PASSWORD")
        mssql_table = 'scorecard_v1'

        logger.info("Environment variables loaded")

        st.cache_data.clear()
        st.cache_resource.clear()
        logger.info("Cleared app cache")

        st.sidebar.image(r"./synoptek.png")

        colored_header(
            label="💬 QA on Agent Calls", description="\n",
            color_name="violet-70",
        )

        with st.sidebar:
            client_name = st.selectbox("**Select client name here 🚩**", ["Mitsui", "Northpoint", "CDI", "Quad-C", "Synoptek Main", "Others"])
            date_ = st.text_input("**Add date here (eg. format: 2025_10_31)** 🚩")
            rfp = st.file_uploader("**Upload Audio File** 🚩", accept_multiple_files=True)

        logger.info("Initialized session state")

        def zip_files_with_structure(files, zip_name, base_dir):
            with zipfile.ZipFile(zip_name, 'w') as zipf:
                for file in files:
                    arcname = os.path.join(base_dir, os.path.basename(file))
                    zipf.write(file, arcname)

        def upload_to_azure_blob_storage(connection_string, container_name, file, relative_path):
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=relative_path)
                
                # Check if the input is a file-like object or a file path
                if hasattr(file, 'read'):
                    data = file.read()
                else:
                    with open(file, "rb") as f:
                        data = f.read()
                
                blob_client.upload_blob(data, overwrite=True)
                logger.info(f"Uploaded {relative_path} to Azure Blob Storage.")
            except Exception as e:
                logger.exception("Error uploading file to Azure Blob Storage: %s", e)
                raise e

        def download_and_unzip_blob(connection_string, container_name, blob_name, extract_to):
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                with open("temp.zip", "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                
                with zipfile.ZipFile("temp.zip", 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if not member.endswith("/"):
                            member_path = os.path.join(extract_to, os.path.basename(member))
                            with open(member_path, 'wb') as output_file:
                                output_file.write(zip_ref.read(member))

                os.remove("temp.zip")
                logger.info(f"Downloaded and extracted {blob_name} to {extract_to}.")
            except Exception as e:
                logger.exception("Error downloading and unzipping blob from Azure Blob Storage: %s", e)
                raise e

        def generate_sharepoint_link(file_name, base_url):
            return f"{base_url}/{file_name}".replace("/.", "")


        # Ensure that client_name and date_ are not empty
        if client_name and date_ and rfp:
            local_folder_path = os.path.join(local_folder, client_name, date_)
            if not os.path.exists(local_folder_path):
                os.makedirs(local_folder_path)

            uploaded_files = []
            for file in rfp:
                file_name = file.name
                file_path = os.path.join(local_folder_path, file_name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                uploaded_files.append(file_path)

            # Zip the uploaded files with the correct structure
            zip_name = os.path.join(local_folder, f"{client_name}.zip")
            zip_files_with_structure(uploaded_files, zip_name, date_)

            # Upload the zipped file to Azure Blob Storage in the correct directory
            blob_path = os.path.join(f"{client_name}_input.zip")
            upload_to_azure_blob_storage(azure_blob_connection_string, azure_blob_container_name, zip_name, blob_path)

            # Download and unzip the file for processing
            download_and_unzip_blob(azure_blob_connection_string, azure_blob_container_name, blob_path, local_folder_path)

            @st.cache_resource
            def azure_openai_setup(azure_openai_api_key, azure_endpoint):
                try:
                    logger.info("Setting up Azure OpenAI")
                    deployment_gpt4o_azure = 'qa-call-agent-gpt4o'
                    llm_azure_resp = AzureChatOpenAI(
                        model_name=deployment_gpt4o_azure,
                        openai_api_key=azure_openai_api_key,
                        azure_endpoint=azure_endpoint,
                        openai_api_version="2024-04-01-preview",
                        temperature=0,
                        max_tokens=4000,
                        model_kwargs={'seed': 123}
                    )
                    logger.info("Azure OpenAI setup completed")
                    return llm_azure_resp
                except Exception as e:
                    logger.exception("Error setting up Azure OpenAI: %s", e)
                    raise e

            llm_azure_resp = azure_openai_setup(azure_openai_api_key, azure_endpoint)

            client = AzureOpenAI(
                api_key=azure_openai_api_key,
                api_version="2024-02-01",
                azure_endpoint=azure_endpoint
            )

            def transcribe_audio(file_path):
                with open(file_path, "rb") as audio_file:
                    result = client.audio.transcriptions.create(
                        file=audio_file,
                        model="qa-call-agent-dev-01"
                    )
                transcript = result.text
                relative_path = os.path.relpath(file_path, local_folder)
                transcription_file = os.path.join(output_folder, relative_path).replace(".wav", ".txt")
                os.makedirs(os.path.dirname(transcription_file), exist_ok=True)
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                return transcript
            
            def evaluate_call(llm_azure_resp, transcript, template):
                prompt = PromptTemplate(template=template, input_variables=["transcript"])
                response = llm_azure_resp.invoke(prompt.format(transcript=transcript))
                evaluation = response.content.strip()
                
                return evaluation
                

            def extract_scorecard_from_evaluation(evaluation_text):
                criteria_pattern = re.compile(r'"Criteria":\s*"([^"]+)"')
                scale_pattern = re.compile(r'"Scale":\s*"([^"]+)"')
                score_pattern = re.compile(r'"Score":\s*(?:"([^"]+)"|(\d+))')
                notes_pattern = re.compile(r'"Notes":\s*"([^"]+)"')
                criteria = criteria_pattern.findall(evaluation_text)
                scales = scale_pattern.findall(evaluation_text)
                scores = [s1 if s1 else s2 for s1, s2 in score_pattern.findall(evaluation_text)]
                notes = notes_pattern.findall(evaluation_text)
                notes = [note.replace('\n', ' ').strip() for note in notes]
                if len(criteria) == len(scales) == len(scores) == len(notes):
                    return pd.DataFrame({
                        "Criteria": criteria,
                        "Scale": scales,
                        "Score": scores,
                        "Notes": notes
                    })
                else:
                    raise ValueError("Mismatch in lengths of extracted fields. Check the regex patterns and evaluation text format.")
            
            def extract_agent_email_id(file_name):
                print(f"Extracting email from file name: {file_name}")
                email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
                match = email_pattern.search(file_name)
                if match:
                    email = match.group(0)
                    print(f"Extracted email: {email}")
                    return email
                else:
                    print("No email found")
                    return "NA"
            
            def format_scorecard_to_expected(df, file_name, transcript_location, score_location):
                expected_columns = [
                    'filename', 'transcript location', 'score location', 'agent email id',
                    'proper greeting 0-1', 'proper greeting notes',
                    'confirmation of contact and company 0-2', 'confirmation of contact and company notes',
                    'demonstrated active listening 0-5', 'demonstrated active listening notes',
                    'problem resolution 0-10', 'problem resolution notes',
                    'remote support 0-1', 'remote support notes',
                    'professionalism and courtesy 0-2', 'professionalism and courtesy notes',
                    'closing 0-5', 'closing notes',
                    'case number provision 0-1', 'case number provision notes'
                ]

                agent_email_id = extract_agent_email_id(file_name)
                print(f"Agent email id for {file_name}: {agent_email_id}")
                
                row_data = {
                    'id': '',  # Fill with appropriate id if available
                    'filename': file_name,
                    'transcript location': transcript_location,
                    'score location': score_location,
                    'agent email id': extract_agent_email_id(file_name)
                }

                for index, row in df.iterrows():
                    if 'Proper Greeting' in row['Criteria']:
                        row_data['proper greeting 0-1'] = row['Score']
                        row_data['proper greeting notes'] = row['Notes']
                    elif 'Confirmation of Contact and Company' in row['Criteria']:
                        row_data['confirmation of contact and company 0-2'] = row['Score']
                        row_data['confirmation of contact and company notes'] = row['Notes']
                    elif 'Demonstrated Active Listening' in row['Criteria']:
                        row_data['demonstrated active listening 0-5'] = row['Score']
                        row_data['demonstrated active listening notes'] = row['Notes']
                    elif 'Problem Resolution' in row['Criteria']:
                        row_data['problem resolution 0-10'] = row['Score']
                        row_data['problem resolution notes'] = row['Notes']
                    elif 'Remote Support' in row['Criteria']:
                        row_data['remote support 0-1'] = row['Score']
                        row_data['remote support notes'] = row['Notes']
                    elif 'Professionalism and Courtesy' in row['Criteria']:
                        row_data['professionalism and courtesy 0-2'] = row['Score']
                        row_data['professionalism and courtesy notes'] = row['Notes']
                    elif 'Closing' in row['Criteria']:
                        row_data['closing 0-5'] = row['Score']
                        row_data['closing notes'] = row['Notes']
                    elif 'Case Number Provision' in row['Criteria']:
                        row_data['case number provision 0-1'] = row['Score']
                        row_data['case number provision notes'] = row['Notes']

                formatted_df = pd.DataFrame([row_data], columns=expected_columns)
                logger.info(f"Formatted DataFrame columns: {formatted_df.columns.tolist()}")
                print('formatted_df',formatted_df)
                formatted_df.to_csv('output.csv',index=False)
                return formatted_df
                

            def upload_csv_to_mssql(transformed_df, server, database, username, password, table, SHAREPOINT_LINK):
                conn = pyodbc.connect(
                    f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Connection Timeout=30')
                cursor = conn.cursor()

                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}')
                CREATE TABLE {table} (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    filename_ VARCHAR(255),
                    transcript_location VARCHAR(MAX),
                    scorefile_location VARCHAR(MAX),
                    agent_email_id VARCHAR(MAX),
                    proper_greeting_0_1 INT,
                    proper_greeting_notes VARCHAR(MAX),
                    confirmation_of_contact_and_company_0_2 INT,
                    confirmation_of_contact_and_company_notes VARCHAR(MAX),
                    demonstrated_active_listening_0_5 INT,
                    demonstrated_active_listening_notes VARCHAR(MAX),
                    problem_resolution_0_10 INT,
                    problem_resolution_notes VARCHAR(MAX),
                    remote_support_0_1 INT,
                    remote_support_notes VARCHAR(MAX),
                    professionalism_and_courtesy_0_2 INT,
                    professionalism_and_courtesy_notes VARCHAR(MAX),
                    closing_0_5 INT,
                    closing_notes VARCHAR(MAX),
                    case_number_provision_0_1 INT,
                    case_number_provision_notes VARCHAR(MAX),
                    created_at DATETIME2 DEFAULT SYSDATETIME()
                );
                """)
                conn.commit()

                for index, row in transformed_df.iterrows():
                    # Base paths for output files and audio files
                    base_output_path = f"qa_call_analysis_output_files/{client_name}_output"
                    base_audio_path = f"qa_call_analysis_audio_files/{client_name}_input"

                    # Original file path
                    original_file_path = row['filename_']
                    print('original_file_path', original_file_path)

                    # Generate paths by removing the unwanted part
                    relative_file_path = original_file_path.replace(f'/qa_call_analysis_audio_files\\{client_name}', '')
                    relative_file_path = relative_file_path.replace('\\', '/')
                    print('relative_file_path', relative_file_path)

                    filename_link = generate_sharepoint_link(f"{base_audio_path}/{relative_file_path}", SHAREPOINT_LINK)
                    print('filename_link', filename_link)

                    relative_transcript_path = original_file_path.replace('.wav', '.txt').replace(f'/qa_call_analysis_audio_files\\{client_name}', '')
                    relative_transcript_path = relative_file_path.replace('\\', '/')
                    transcript_link = generate_sharepoint_link(f"{base_output_path}/{relative_transcript_path}", SHAREPOINT_LINK)
                    print('transcript_link', transcript_link)

                    relative_score_path = original_file_path.replace('.wav', '.csv').replace(f'/qa_call_analysis_audio_files\\{client_name}', '')
                    relative_score_path = relative_score_path.replace('\\', '/')
                    score_link = generate_sharepoint_link(f"{base_output_path}/{relative_score_path}", SHAREPOINT_LINK)
                    print('score_link', score_link)

                    cursor.execute(f"""
                    IF EXISTS (SELECT 1 FROM {table} WHERE filename_ = ?)
                    BEGIN
                        UPDATE {table}
                        SET 
                            filename_ = ?,
                            transcript_location = ?, 
                            scorefile_location = ?, 
                            agent_email_id = ?,
                            proper_greeting_0_1 = ?, 
                            proper_greeting_notes = ?, 
                            confirmation_of_contact_and_company_0_2 = ?, 
                            confirmation_of_contact_and_company_notes = ?, 
                            demonstrated_active_listening_0_5 = ?, 
                            demonstrated_active_listening_notes = ?, 
                            problem_resolution_0_10 = ?, 
                            problem_resolution_notes = ?, 
                            remote_support_0_1 = ?, 
                            remote_support_notes = ?, 
                            professionalism_and_courtesy_0_2 = ?, 
                            professionalism_and_courtesy_notes = ?, 
                            closing_0_5 = ?, 
                            closing_notes = ?, 
                            case_number_provision_0_1 = ?, 
                            case_number_provision_notes = ?
                        WHERE filename_ = ?
                    END
                    ELSE
                    BEGIN
                        INSERT INTO {table} (
                            filename_,
                            transcript_location,
                            scorefile_location,
                            agent_email_id,
                            proper_greeting_0_1, 
                            proper_greeting_notes, 
                            confirmation_of_contact_and_company_0_2, 
                            confirmation_of_contact_and_company_notes, 
                            demonstrated_active_listening_0_5, 
                            demonstrated_active_listening_notes, 
                            problem_resolution_0_10, 
                            problem_resolution_notes, 
                            remote_support_0_1, 
                            remote_support_notes, 
                            professionalism_and_courtesy_0_2, 
                            professionalism_and_courtesy_notes, 
                            closing_0_5, 
                            closing_notes, 
                            case_number_provision_0_1, 
                            case_number_provision_notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    END
                    """,
                    row['filename_'],
                    filename_link,
                    transcript_link,
                    score_link,
                    row['agent_email_id'],
                    row['proper_greeting_0_1'],
                    row['proper_greeting_notes'],
                    row['confirmation_of_contact_and_company_0_2'],
                    row['confirmation_of_contact_and_company_notes'],
                    row['demonstrated_active_listening_0_5'],
                    row['demonstrated_active_listening_notes'],
                    row['problem_resolution_0_10'],
                    row['problem_resolution_notes'],
                    row['remote_support_0_1'],
                    row['remote_support_notes'],
                    row['professionalism_and_courtesy_0_2'],
                    row['professionalism_and_courtesy_notes'],
                    row['closing_0_5'],
                    row['closing_notes'],
                    row['case_number_provision_0_1'],
                    row['case_number_provision_notes'],
                    row['filename_'],  # Parameter for WHERE clause in UPDATE
                    filename_link,
                    transcript_link,
                    score_link,
                    row['agent_email_id'],
                    row['proper_greeting_0_1'],
                    row['proper_greeting_notes'],
                    row['confirmation_of_contact_and_company_0_2'],
                    row['confirmation_of_contact_and_company_notes'],
                    row['demonstrated_active_listening_0_5'],
                    row['demonstrated_active_listening_notes'],
                    row['problem_resolution_0_10'],
                    row['problem_resolution_notes'],
                    row['remote_support_0_1'],
                    row['remote_support_notes'],
                    row['professionalism_and_courtesy_0_2'],
                    row['professionalism_and_courtesy_notes'],
                    row['closing_0_5'],
                    row['closing_notes'],
                    row['case_number_provision_0_1'],
                    row['case_number_provision_notes']
                    )
                    conn.commit()

                cursor.close()
                conn.close()
                logger.info(f"Uploaded transformed DataFrame to MSSQL table {table}.")


            template = """
        Given a transcript of a customer support call, evaluate the performance of the support representative based on the following criteria. Each criterion should be rated on a scale provided with each question. from 0 to 5, where 0 indicates the criterion was not met at all, and 5 indicates the criterion was fully met. Additionally, provide specific comments or notes for each criterion to justify the given score.

        1) Proper Greeting (rate on a scale from 0 to 1, where 0 indicates the criterion was not met at all, and 1 indicates the criterion was met): Did the representative start the call with a professional and courteous greeting? For example, "Thank you for calling [Synoptek or Service Desk], My name is [name], how can I help you?

        2) Confirmation of Contact and Company (rate on a scale from 0 to 2, where 0 indicates the criterion was not met at all, and 2 indicates the criterion was fully met): Did the representative explicitly ask about the caller's name/identity? Did the representative explicitly ask for the company name the caller is representing or calling from? The representative MUST ask for the company name and receive it from the caller in order to get credit.

        3) Demonstrated Active Listening (rate on a scale from 0 to 5, where 0 indicates the criterion was not met at all, and 5 indicates the criterion was fully met): Did the representative demonstrate active listening by restating or summarizing the caller's issues or concerns to ensure understanding? Did the representative ask relevant follow-up and troubleshooting questions that build on what the customer has already said?

        4) Problem Resolution (rate on a scale from 0 to 10, where 0 indicates the criterion was not met at all, and 10 indicates the criterion was fully met): Was the representative able to address and resolve the caller's issue during the call, or provide a clear path toward resolution? Did the representative perform relevant troubleshooting steps?

        5) Remote Support (if remote connection was not needed, put "N/A" for the score; if remote connection was needed, rate on a scale from 0 to 1, where 0 indicates permission was not obtained, and 1 indicates permission was obtained): Did the representative need to remotely connect to the caller's system in order to provide assistance? If so, did the representative verbally obtain explicit permission from the caller before connecting to their system? For example, did the representative ask "Do I have your permission to connect to your system remotely?" or something similar? To be clear, implied permission is NOT ENOUGH to get a score of 1. The representative MUST explicitly ask for permission and receive verbal affirmation from the caller!

        6) Professionalism and Courtesy (rate on a scale from 0 to 2, where 0 indicates the criterion was not met at all, and 2 indicates the criterion was fully met): Did the representative maintain a professional and courteous demeanor throughout the call? Did the agent explain any holds/dead air in advance or avoided holds/dead air entirely?

        7) Closing (rate on a scale from 0 to 5, where 0 indicates the criterion was not met at all, and 5 indicates the criterion was fully met): Did the representative end the call on a positive note, confirming that the caller's concerns were addressed and asking if there was anything else they could help with? Did the representative explicitly ask permission to close the case? (this last question is worth 2 points in the score)

        8) Did the representative verbally provide the caller with an explicit case number (rate on a scale from 0 to 1, where 0 indicates the criterion was not met at all, and 1 indicates the criterion was met)? If a verbal case number was provided, comment in the Notes as to when in the call the case number was provided by a percentage of the call duration. It should occur within the first 2 minutes of the call.

        For each criterion, provide a score based on the transcript, followed by notes that include direct quotes or observations from the call to support your rating. Your output should summarize the representative's performance in a structured and detailed manner, similar to a scorecard, and provide a table in json format that contains the questions and answers. The table should have 4 columns: Criteria, Scale, Score, and Notes.
        ---
        Here is the transcript:\n{transcript}
        """

            if __name__ == "__main__":
                all_files = [os.path.join(root, file) for root, _, files in os.walk(local_folder_path) for file in files]

                evaluation_results = []

                for file in all_files:
                    if file.endswith(".wav"):
                        transcript = transcribe_audio(file)
                        evaluation = evaluate_call(llm_azure_resp, transcript, template)
                        scores_and_notes_df = extract_scorecard_from_evaluation(evaluation)
                        scores_and_notes_df['agent_email_id'] = extract_agent_email_id(file)
                        st.write("**Scores and Notes Table:**", file.split("\\")[-1])
                        st.dataframe(scores_and_notes_df)
                        evaluation_results.append(scores_and_notes_df)

                        relative_path = os.path.relpath(file, local_folder_path)
                        output_relative_path = os.path.join(output_folder, relative_path)
                        csv_file_name = output_relative_path.replace(".wav", ".csv")
                        os.makedirs(os.path.dirname(csv_file_name), exist_ok=True)
                        scores_and_notes_df.to_csv(csv_file_name, index=False)

                        transcription_output_file = output_relative_path.replace(".wav", ".txt")
                        os.makedirs(os.path.dirname(transcription_output_file), exist_ok=True)
                        with open(transcription_output_file, 'w', encoding='utf-8') as f:
                            f.write(transcript)

                        formatted_scorecard_df = format_scorecard_to_expected(scores_and_notes_df, file, transcription_output_file, csv_file_name)
                        formatted_scorecard_df.columns = [
                            'filename_', 'transcript_location', 'scorefile_location','agent_email_id', 'proper_greeting_0_1',
                            'proper_greeting_notes', 'confirmation_of_contact_and_company_0_2',
                            'confirmation_of_contact_and_company_notes', 'demonstrated_active_listening_0_5',
                            'demonstrated_active_listening_notes', 'problem_resolution_0_10',
                            'problem_resolution_notes', 'remote_support_0_1', 'remote_support_notes',
                            'professionalism_and_courtesy_0_2', 'professionalism_and_courtesy_notes',
                            'closing_0_5', 'closing_notes', 'case_number_provision_0_1', 'case_number_provision_notes'
                        ]
                        formatted_scorecard_df['case_number_provision_0_1'].fillna(-1, inplace=True)
                        formatted_scorecard_df['case_number_provision_notes'].fillna(' ', inplace=True)
                        formatted_scorecard_df['remote_support_0_1'].replace('N/A', -1, inplace=True)

                        upload_csv_to_mssql(formatted_scorecard_df, mssql_server, mssql_database, mssql_username, mssql_password, mssql_table, SHAREPOINT_LINK)

                def collect_output_files(output_folder):
                    output_files = []
                    for root, dirs, files in os.walk(output_folder):
                        for file in files:
                            if file.endswith(".csv") or file.endswith(".txt"):
                                output_files.append(os.path.join(root, file))
                    return output_files

                output_files = collect_output_files(output_folder)
                output_zip_name = os.path.join(output_folder, f"{client_name}_output.zip")
                zip_files_with_structure(output_files, output_zip_name, date_)

                output_blob_path = os.path.join(f"{client_name}_output.zip")
                upload_to_azure_blob_storage(azure_blob_connection_string, azure_blob_container_name, output_zip_name, output_blob_path)

            with st.sidebar:
                st.sidebar.markdown("""<div style="height: 6vh;"></div>""", unsafe_allow_html=True)
                st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
                authenticator.logout('Logout', 'sidebar')

else:
    if st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')




