# QA-on-Call-Agent
This Streamlit application provides an automated solution for evaluating the performance of support representatives based on the analysis of their call transcripts. It integrates several technologies including Azure Blob Storage, Azure OpenAI, MSSQL, and two-factor authentication for secure access.

Features
Authentication and Two-Factor Authentication (2FA):

User authentication using credentials stored in an Azure Blob Storage.
Two-factor authentication setup using OTP (One Time Password) to enhance security.
File Upload and Processing:

Users can upload audio files in .wav format for analysis.
Uploaded files are stored locally and then zipped before being uploaded to Azure Blob Storage.
Audio Transcription:

Audio files are transcribed using Azure OpenAI's transcription service.
Transcriptions are saved locally in .txt format.
Performance Evaluation:

Transcripts are evaluated using an Azure OpenAI language model.
Evaluation is based on specific criteria including greeting, confirmation of contact, active listening, problem resolution, remote support, professionalism, and closing.
Scores and notes for each criterion are extracted and formatted into a structured DataFrame.
Data Storage:

Evaluation results are formatted to match a predefined schema and uploaded to a MSSQL database.
Links to the original audio, transcription, and evaluation scorecard files are generated for easy access.
Output Management:

Evaluation results and transcriptions are saved as CSV and TXT files respectively.
These files are zipped and uploaded back to Azure Blob Storage.
User Interface:

A sidebar for user interaction to select client name, input date, and upload files.
Display of evaluation results in a structured table format within the app.
Setup Instructions
Prerequisites
Python 3.7 or higher
Azure Blob Storage account
MSSQL database
Azure OpenAI subscription
Required Python packages (listed in requirements.txt)
