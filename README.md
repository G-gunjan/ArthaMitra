

SmartDesk is a prototype banking customer-support assistant. It combines
a fine-tuned TinyBERT intent classifier, DistilBERT sentiment analysis,
rule-based banking issue and priority detection, FAQ recommendations,
prediction history, CSV export, and an Ollama-powered Qwen chatbot.

## 🎥 Project Demo

Watch the working demo of ArthaMitra – Banking AI Assistant:

https://drive.google.com/file/d/1kxq60woYySWunxuYXFEDX14HYuwZ9J9O/view?usp=sharing
------------------------------------------------------------------------

## Table of contents

-   [Product overview](#product-overview)
-   [Product features](#product-features)
-   [User workflow](#user-workflow)
-   [System architecture](#system-architecture)
-   [Machine-learning details](#machine-learning-details)
-   [Current evaluation results](#current-evaluation-results)
-   [Technology stack](#technology-stack)
-   [Project setup](#project-setup)
-   [Running the application](#running-the-application)
-   [How to use](#how-to-use)
-   [Testing and test cases](#testing-and-test-cases)
-   [Product backlog](#product-backlog)
-   [Known limitations and risks](#known-limitations-and-risks)
-   [Security and responsible use](#security-and-responsible-use)
-   [Suggested repository structure](#suggested-repository-structure)

------------------------------------------------------------------------

## Product overview

### Problem statement

Banking customers may describe the same issue in different ways and
languages. Support teams need to understand the customer's request,
identify urgent cases, and provide clear next steps. Manually reviewing
every message can be time-consuming.

SmartDesk demonstrates an AI-assisted workflow that:

1.  Accepts a banking query in English, Hindi, or Marathi.
2.  Translates supported Hindi/Marathi messages into English for
    downstream models.
3.  Predicts one of the Banking77 intent categories.
4.  Estimates sentiment using a separate sentiment model.
5.  Detects selected banking issues and complaint priority using rules.
6.  Recommends a relevant FAQ when confidence is sufficient.
7.  Uses Qwen through Ollama to draft a helpful response for non-fraud
    queries.
8.  Uses fixed safety guidance for suspected fraud instead of sending
    those cases to the chatbot model.

### Product goals

-   Make banking-query triage easier to understand.
-   Provide a simple human-review signal for uncertain predictions.
-   Support common English, Hindi, and Marathi banking phrases.
-   Keep suspected fraud handling conservative and safety-focused.
-   Demonstrate how traditional ML, transformer models, rules, and an
    LLM can work together.

### Intended users

-   Banking customer-support trainees and demonstrators.
-   Project evaluators exploring NLP-based customer-query
    classification.
-   Developers extending a prototype into a more complete support
    workflow.

This is an academic/prototype application, not a bank-operated service
or a replacement for official bank support.

------------------------------------------------------------------------

## Product features

  ---------------------------------------------------------------------------------------------------
  Feature                 Description                                         Current implementation
  ----------------------- --------------------------------------------------- -----------------------
  Banking intent          Predicts a Banking77 intent from a customer message Implemented in supplied
  classification          using fine-tuned TinyBERT                           code

  Top-5 predictions       Displays the five highest-scoring intent labels and Implemented
                          confidence values                                   

  Multilingual input      Handles English and includes Hindi/Marathi phrase   Partially implemented;
                          mappings and language detection                     see limitations

  English translation     Uses direct phrase mappings for known expressions   Implemented
                          and Qwen for other detected Hindi/Marathi messages  

  Sentiment analysis      Uses                                                Implemented
                          `distilbert-base-uncased-finetuned-sst-2-english`   
                          to estimate positive/negative sentiment             

  Banking issue detection Applies rules for suspected fraud, blocked          Implemented
                          cards/accounts, ATM cash issues, debits, and failed 
                          payments                                            

  Complaint priority      Assigns Critical, High, or Normal priority using    Implemented
                          fraud and issue rules                               

  Human-review indicator  Compares intent confidence with a configurable      Implemented
                          threshold                                           

  FAQ recommendations     Selects a FAQ based on the predicted intent or      Implemented
                          broad intent keywords                               

  AI chatbot              Uses local Qwen 2.5 3B through Ollama to generate   Implemented; requires
                          practical responses                                 Ollama and model

  Fraud-safe response     Uses predefined English, Hindi, or Marathi guidance Implemented
                          for suspected fraud                                 

  Prediction history      Keeps predictions in Streamlit session state        Implemented;
                                                                              session-only

  CSV export              Downloads prediction history as a CSV file          Implemented

  Clear history           Clears the current session's prediction history     Implemented
  ---------------------------------------------------------------------------------------------------

### Main application tabs

1.  **Intent Classifier** --- enter a message or choose an example,
    classify it, and inspect intent, confidence, sentiment, issue,
    priority, FAQ, and top-5 predictions.
2.  **Prediction History** --- review predictions created in the current
    Streamlit session, download them as CSV, or clear the list.
3.  **AI Banking Chatbot** --- ask banking questions and choose
    automatic language detection or English, Hindi, or Marathi
    responses.

------------------------------------------------------------------------

## User workflow

``` text
Customer message
       |
       v
Language detection / known phrase mapping
       |
       v
English translation (when required)
       |
       +------------------------------+
       |                              |
       v                              v
TinyBERT intent prediction      DistilBERT sentiment
       |                              |
       +---------------+--------------+
                       |
                       v
        Rule-based issue and priority detection
                       |
                       v
             Confidence threshold check
                       |
             +---------+---------+
             |                   |
       Low confidence       Sufficient confidence
             |                   |
       Human review        FAQ may be recommended
             |                   |
             +---------+---------+
                       |
                       v
        Chat response generation (chatbot tab)
                       |
             +---------+---------+
             |                   |
        Suspected fraud       Other query
             |                   |
      Fixed safe guidance     Qwen via Ollama
             |                   |
             +---------+---------+
                       |
                       v
             Display result to user
```

The classifier and chatbot are separate user-facing workflows in the
supplied Streamlit code. The classifier tab displays analysis results;
the chatbot tab independently performs analysis before generating a
conversational answer.

------------------------------------------------------------------------

## System architecture

### 1. Presentation layer

-   Built with Streamlit.
-   Provides three tabs: Intent Classifier, Prediction History, and AI
    Banking Chatbot.
-   Provides a sidebar human-review threshold slider (0.30--0.95,
    default 0.60).
-   Displays the selected compute device, number of intent labels, model
    names, and Ollama availability.

### 2. NLP preprocessing layer

-   Normalizes text by lowercasing, trimming whitespace, and removing
    selected punctuation.
-   Detects English, Hindi, and Marathi using word/phrase heuristics.
-   Maps a set of known Hindi and Marathi expressions---including
    Roman-script variants---to English.
-   Uses Qwen via Ollama to translate other detected Hindi/Marathi
    messages.
-   Avoids LLM translation for messages detected as suspected fraud and
    uses a neutral English description instead.

### 3. ML inference layer

-   TinyBERT predicts the banking intent.
-   Softmax probabilities are used to obtain the top prediction,
    confidence, and top-five candidates.
-   DistilBERT produces a separate sentiment label and confidence.
-   Models are loaded with Transformers and moved to CUDA when
    available, otherwise CPU.

### 4. Business-rule layer

-   Suspected fraud takes precedence in banking issue detection.
-   Additional keyword rules detect selected card, account, ATM, debit,
    and payment issues.
-   Priority is assigned as Critical, High, or Normal.
-   FAQ selection uses an FAQ dictionary and intent-keyword fallbacks.
-   The confidence threshold determines whether the intent is marked for
    human review.

### 5. Response-generation layer

-   The chatbot uses the local `qwen2.5:3b` model through Ollama.
-   The system prompt asks for concise, practical steps and prohibits
    invented bank policies, phone numbers, transaction details, refund
    promises, or claims of account access.
-   Recent chat context is included in the prompt.
-   Suspected fraud bypasses Qwen response generation and uses a fixed
    language-specific safety response.

### 6. Session and export layer

-   Prediction history and chat messages are held in `st.session_state`.
-   Prediction history can be downloaded as UTF-8 CSV.
-   The supplied code does not show a database or permanent history
    store.

------------------------------------------------------------------------

## Machine-learning details

### Banking intent classifier

  Item                                    Value
  --------------------------------------- ------------------------------------------------
  Dataset                                 `PolyAI/banking77`
  Training examples                       10,003
  Official test examples                  3,080
  Intent classes                          77
  Validation split                        10% of the training split, stratified by label
  Training split after validation split   9,002
  Validation examples                     1,001
  Pretrained checkpoint                   `huawei-noah/TinyBERT_General_4L_312D`
  Maximum token length                    128
  Random seed                             42
  Learning rate                           `2e-5`
  Training epochs                         4
  Training batch size                     16
  Evaluation batch size                   32
  Weight decay                            0.01
  Best-model metric                       Macro F1

The notebook uses Hugging Face `datasets` and `transformers`, tokenizes
text with the checkpoint tokenizer, dynamically pads batches, and
fine-tunes a sequence-classification head for 77 labels.

### Baseline model

The notebook also evaluates a TF-IDF + Logistic Regression baseline
using word unigrams/bigrams and a Logistic Regression classifier. This
is important because the measured baseline performs substantially better
than the TinyBERT run shown in the notebook.

### Sentiment model

-   Model: `distilbert-base-uncased-finetuned-sst-2-english`
-   Task: English positive/negative sentiment classification.
-   The result is an estimate and should not be interpreted as a
    definitive measure of a customer's emotional state.
-   Hindi and Marathi text is translated to English before sentiment
    analysis in the supplied workflow.

### Chatbot model

-   Runtime: Ollama
-   Model tag configured in the app: `qwen2.5:3b`
-   Used for non-fraud response generation and, where needed,
    Hindi/Marathi-to-English translation.
-   It runs locally through Ollama; it is not a hosted banking API.

------------------------------------------------------------------------

## Current evaluation results

The supplied notebook reports the following test-set results:

  Model                            Accuracy   Macro Precision   Macro Recall   Macro F1
  ------------------------------ ---------- ----------------- -------------- ----------
  TF-IDF + Logistic Regression       0.8588            0.8681         0.8588     0.8581
  TinyBERT                           0.1597            0.1407         0.1597     0.0886

**Important:** The TinyBERT result is weak in the supplied experiment.
Do not describe the TinyBERT classifier as production-ready or claim
high accuracy. The TF-IDF + Logistic Regression baseline is the stronger
measured model in this run. Investigate label mapping, preprocessing,
training configuration, checkpoint compatibility, and class-wise errors
before relying on TinyBERT predictions.

The notebook also shows a TinyBERT test loss of `3.5197`. Use fresh
evaluation after any model or preprocessing changes; do not assume these
figures describe a later model artifact.

------------------------------------------------------------------------

## Technology stack

  Area                            Technology
  ------------------------------- ---------------------------
  Language                        Python
  UI                              Streamlit
  Deep learning                   PyTorch
  NLP / pretrained models         Hugging Face Transformers
  Dataset loading                 Hugging Face Datasets
  Intent dataset                  Banking77
  Intent classifier               TinyBERT
  Sentiment classifier            DistilBERT SST-2
  Chatbot / translation runtime   Ollama + Qwen 2.5 3B
  Data handling                   Pandas, NumPy
  Evaluation                      scikit-learn
  Notebook visualisation          Matplotlib, Seaborn

------------------------------------------------------------------------

## Project setup

### Prerequisites

-   Python 3.10 or later is recommended. Use a Python version supported
    by the installed PyTorch and Transformers releases.
-   `pip` and `venv`.
-   Ollama installed and available on the machine.
-   Enough memory and disk space for the downloaded transformer models.
-   The trained TinyBERT model directory expected by the app:
    `./smartdesk_tinybert_model`.

### 1. Clone or open the project

``` bash
git clone <your-repository-url>
cd <your-project-folder>
```

If you already have the project locally, open a terminal in the project
root.

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install application dependencies

``` bash
python -m pip install --upgrade pip
pip install streamlit torch transformers pandas ollama
```

For the training notebook, install the additional packages:

``` bash
pip install datasets accelerate evaluate scikit-learn numpy matplotlib seaborn
```

For reproducible submissions, record the exact working package versions
in a `requirements.txt`. The supplied notebook output includes a
pandas/Google Colab dependency conflict, so avoid blindly upgrading
packages in an existing Colab runtime.

### 4. Install and prepare Ollama

Install Ollama from its official website, then open a terminal and run:

``` bash
ollama pull qwen2.5:3b
ollama list
```

Ensure the Ollama service is running before using chatbot or LLM-based
translation features. The application checks whether the configured
model is available and displays its status in the sidebar.

### 5. Prepare the TinyBERT model directory

The Streamlit code loads the tokenizer and model from:

``` text
./smartdesk_tinybert_model
```

That directory must contain a compatible Hugging Face model and
tokenizer, including the model configuration and weight files. The
training notebook shown in the supplied PDF uses a checkpoint directory
for training; verify that your final trained model is exported to the
exact directory expected by the app.

If exporting from the Hugging Face `Trainer`, the usual export calls
are:

``` python
trainer.save_model("./smartdesk_tinybert_model")
tokenizer.save_pretrained("./smartdesk_tinybert_model")
```

Run these after training and evaluation, then place the folder beside
the Streamlit application file. Confirm that the saved label
configuration matches the 77 Banking77 intent names expected by the
application.

### 6. Start the Streamlit application

If your application script is named `app.py`:

``` bash
streamlit run app.py
```

If the Python file has a different name, replace `app.py` with that
filename. Streamlit prints a local URL (commonly
`http://localhost:8501`) in the terminal.

------------------------------------------------------------------------

## How to use

### Intent Classifier

1.  Open the **Intent Classifier** tab.
2.  Enter a banking query or select one of the provided examples.
3.  Click **Classify Intent**.
4.  Review the predicted intent, confidence, sentiment, rule-based
    issue, complaint priority, and top-five intent predictions.
5.  Review the FAQ recommendation.
6.  If the status says **Needs human review**, do not treat the
    predicted intent as confirmed.

### Prediction History

1.  Classify one or more messages.
2.  Open **Prediction History**.
3.  Review the current session's records.
4.  Select **Download Prediction History (CSV)** to export them.
5.  Select **Clear Prediction History** to clear the session list.

### AI Banking Chatbot

1.  Open **AI Banking Chatbot**.
2.  Choose automatic language detection or a preferred response
    language.
3.  Ask a banking question.
4.  The app analyses the message and generates a response.
5.  Suspected fraud messages receive predefined safety guidance instead
    of an LLM-generated answer.

------------------------------------------------------------------------

## Testing and test cases

The following test cases are a proposed QA checklist based on the
current product behaviour. They are not evidence that the tests have
already been executed. Run them against the actual application and
record the observed result.

### A. Intent classifier and input validation

  ---------------------------------------------------------------------------------------------------------------------
  ID             Test scenario              Test input / action                      Expected result     Priority
  -------------- -------------------------- ---------------------------------------- ------------------- --------------
  TC-01          Valid English query        `How do I activate my new debit card?`   Prediction details  High
                                                                                     are displayed       
                                                                                     without an          
                                                                                     application error   

  TC-02          Valid Hindi phrase         `मेरा डेबिट कार्ड ब्लॉक हो गया है`            Message is accepted High
                                                                                     and                 
                                                                                     mapped/translated   
                                                                                     for analysis        

  TC-03          Valid Marathi phrase       `माझे डेबिट कार्ड ब्लॉक झाले आहे`              Message is accepted High
                                                                                     and                 
                                                                                     mapped/translated   
                                                                                     for analysis        

  TC-04          Roman Hindi phrase         `Mera debit card block ho gaya hai`      Known phrase        High
                                                                                     mapping returns an  
                                                                                     English equivalent  

  TC-05          Roman Marathi phrase       `Majha debit card block jhala ahe`       Known phrase        High
                                                                                     mapping returns an  
                                                                                     English equivalent  

  TC-06          Empty input                Submit without entering a message        Warning asks user   High
                                                                                     to enter a message  

  TC-07          Whitespace-only input      Enter spaces and submit                  Input is rejected   Medium
                                                                                     with validation     
                                                                                     warning             

  TC-08          Long input                 Submit a message longer than 128 tokens  Tokenizer truncates Medium
                                                                                     input; app remains  
                                                                                     responsive          

  TC-09          Punctuation and casing     `MY CARD IS BLOCKED!!!`                  Normalization runs  Medium
                                                                                     and prediction is   
                                                                                     produced            

  TC-10          Unsupported/unrecognised   Enter a language not supported by the    No incorrect        Medium
                 language                   app                                      language-support    
                                                                                     claim; graceful     
                                                                                     handling or clear   
                                                                                     error               

  TC-11          Model directory missing    Temporarily rename model directory and   App displays        High
                                            launch                                   model-loading error 
                                                                                     and stops cleanly   

  TC-12          Top-five output            Submit a valid message                   Up to five intent   Medium
                                                                                     candidates and      
                                                                                     confidence values   
                                                                                     are shown           
  ---------------------------------------------------------------------------------------------------------------------

### B. Confidence and human review

  --------------------------------------------------------------------------------------------
  ID             Test scenario    Test input /     Expected result              Priority
                                  action                                        
  -------------- ---------------- ---------------- ---------------------------- --------------
  TC-13          Confidence above Use a query      Status is                    High
                 threshold        whose confidence `Automatically classified`   
                                  is above the                                  
                                  configured                                    
                                  threshold                                     

  TC-14          Confidence below Use a query      Status is                    High
                 threshold        whose confidence `Needs human review`         
                                  is below the                                  
                                  configured                                    
                                  threshold                                     

  TC-15          Change review    Move threshold   Classification status        High
                 threshold        from 0.60 to     follows the selected         
                                  another allowed  threshold                    
                                  value                                         

  TC-16          Boundary         Test confidence  Verify implementation's `<`  Medium
                 confidence       equal to         comparison; equality is not  
                                  threshold        marked for review            

  TC-17          Low-confidence   Submit a         Safe uncertainty guidance is High
                 FAQ              low-confidence   shown instead of forcing an  
                                  query            unrelated FAQ                
  --------------------------------------------------------------------------------------------

### C. Issue detection and complaint priority

  --------------------------------------------------------------------------------------------------------------------------
  ID             Test scenario  Test input / action                                           Expected result Priority
  -------------- -------------- ------------------------------------------------------------- --------------- --------------
  TC-18          Suspected      `Someone withdrew money from my account without permission`   Issue is        Critical
                 fraud in                                                                     flagged as      
                 English                                                                      possible fraud  
                                                                                              and priority is 
                                                                                              Critical        

  TC-19          Suspected      `मेरे खाते से किसी ने पैसे निकाल लिए`                                Fraud rule is   Critical
                 fraud in Hindi                                                               triggered and   
                                                                                              fixed Hindi     
                                                                                              safety response 
                                                                                              is used in      
                                                                                              chatbot         

  TC-20          Suspected      `माझ्या खात्यातून कोणीतरी पैसे काढले`                               Fraud rule is   Critical
                 fraud in                                                                     triggered and   
                 Marathi                                                                      fixed Marathi   
                                                                                              safety response 
                                                                                              is used in      
                                                                                              chatbot         

  TC-21          Blocked card   `My debit card is blocked`                                    Blocked-card    High
                                                                                              issue is        
                                                                                              detected;       
                                                                                              priority is     
                                                                                              High            

  TC-22          Frozen account `My bank account is frozen`                                   Account issue   High
                                                                                              is detected;    
                                                                                              priority is     
                                                                                              High            

  TC-23          ATM cash not   `Money was deducted but cash was not dispensed`               ATM issue is    High
                 received                                                                     detected;       
                                                                                              priority is     
                                                                                              High            

  TC-24          Money deducted `Money has been deducted from my account`                     Account debit   High
                                                                                              issue is        
                                                                                              detected;       
                                                                                              priority is     
                                                                                              High            

  TC-25          Failed         `My UPI payment failed`                                       Payment issue   High
                 UPI/payment                                                                  is detected;    
                                                                                              priority is     
                                                                                              High            

  TC-26          General query  `How can I update my address?`                                No false        Medium
                                                                                              critical/high   
                                                                                              rule should be  
                                                                                              triggered;      
                                                                                              priority        
                                                                                              remains Normal  
                                                                                              unless another  
                                                                                              rule applies    
  --------------------------------------------------------------------------------------------------------------------------

### D. Sentiment analysis and FAQ

  -------------------------------------------------------------------------------------------------------------
  ID             Test scenario  Test input / action                               Expected       Priority
                                                                                  result         
  -------------- -------------- ------------------------------------------------- -------------- --------------
  TC-27          Positive       `Thank you, the issue is resolved`                Sentiment      Medium
                 English text                                                     label and      
                                                                                  confidence are 
                                                                                  displayed      

  TC-28          Negative       `I am very frustrated with this failed payment`   Sentiment      Medium
                 English text                                                     label and      
                                                                                  confidence are 
                                                                                  displayed      

  TC-29          Sentiment      Use a message with a low sentiment                Result is      Low
                 uncertainty    score/confidence                                  displayed as   
                                                                                  returned by    
                                                                                  the sentiment  
                                                                                  model; avoid   
                                                                                  treating it as 
                                                                                  definitive     

  TC-30          FAQ for card   Submit a card-related intent                      Card FAQ or    Medium
                 intent                                                           appropriate    
                                                                                  fallback is    
                                                                                  displayed      

  TC-31          FAQ for        Submit a transfer/beneficiary intent              Transfer FAQ   Medium
                 transfer                                                         or appropriate 
                 intent                                                           fallback is    
                                                                                  displayed      

  TC-32          Unknown FAQ    Submit an intent without a direct FAQ entry       General        Medium
                 intent                                                           banking        
                                                                                  assistance     
                                                                                  fallback is    
                                                                                  displayed      
  -------------------------------------------------------------------------------------------------------------

### E. Chatbot and language handling

  ----------------------------------------------------------------------------------
  ID             Test scenario  Test input /    Expected result       Priority
                                action                                
  -------------- -------------- --------------- --------------------- --------------
  TC-33          English        Ask a normal    Qwen returns a        High
                 chatbot query  banking         concise practical     
                                question in     response              
                                English                               

  TC-34          Hindi chatbot  Ask a non-fraud Response follows      High
                 query          question in     selected/detected     
                                Hindi           Hindi language        

  TC-35          Marathi        Ask a non-fraud Response follows      High
                 chatbot query  question in     selected/detected     
                                Marathi         Marathi language      

  TC-36          Explicit       Select English  Response is requested Medium
                 response       and enter a     in English            
                 language       Hindi query                           

  TC-37          Fraud chatbot  Ask about an    Predefined fraud      Critical
                 bypass         unauthorised    response is returned; 
                                withdrawal      Qwen is bypassed for  
                                                advice                

  TC-38          Ollama         Stop Ollama and A clear error is      High
                 unavailable    send a normal   shown; app does not   
                                chatbot query   silently claim a      
                                                response was          
                                                generated             

  TC-39          Model not      Remove/rename   Sidebar reports model High
                 installed      `qwen2.5:3b`    unavailable           
                                and reload                            

  TC-40          Empty model    Simulate empty  Application reports a Medium
                 response       Qwen output     response-generation   
                                                error                 

  TC-41          Chat context   Send multiple   Recent conversation   Medium
                                related         context is included   
                                messages        in the next model     
                                                request               

  TC-42          Confidential   Enter an        Assistant does not    Critical
                 information    OTP/PIN-like    request or repeat     
                                string in a     sensitive credentials 
                                test            unnecessarily; review 
                                environment     logging/privacy       
                                                behaviour             
  ----------------------------------------------------------------------------------

### F. History, export, and application behaviour

  ------------------------------------------------------------------------------
  ID             Test scenario  Test input /      Expected result Priority
                                action                            
  -------------- -------------- ----------------- --------------- --------------
  TC-43          Add prediction Classify a valid  A record is     High
                 to history     query             added to        
                                                  session         
                                                  prediction      
                                                  history         

  TC-44          Empty history  Open history      Empty-state     Medium
                                before            message is      
                                classifying       displayed       
                                anything                          

  TC-45          CSV download   Download history  CSV downloads   High
                                after multiple    and contains    
                                predictions       the displayed   
                                                  records         

  TC-46          CSV Unicode    Export records    UTF-8 CSV       High
                                containing        preserves text  
                                Hindi/Marathi     correctly       
                                text                              

  TC-47          Clear history  Click Clear       Current session High
                                Prediction        history becomes 
                                History           empty           

  TC-48          Session        Refresh/restart   Verify expected Medium
                 refresh        the app           session-state   
                                                  reset; history  
                                                  is not promised 
                                                  to persist      

  TC-49          CPU execution  Run with CUDA     Models run on   High
                                unavailable       CPU without     
                                                  CUDA-only       
                                                  operations      

  TC-50          GPU execution  Run where CUDA is Models are      Medium
                                available         moved to CUDA   
                                                  and inference   
                                                  completes       
  ------------------------------------------------------------------------------

### Test execution record

Use this table during manual QA or a project demonstration.

  --------------------------------------------------------------------------
  Test ID        Actual result  Pass/Fail      Evidence /     Tester / date
                                               defect ID      
  -------------- -------------- -------------- -------------- --------------
  TC-01                         Not run                       

  TC-06                         Not run                       

  TC-18                         Not run                       

  TC-33                         Not run                       

  TC-37                         Not run                       

  TC-45                         Not run                       
  --------------------------------------------------------------------------

------------------------------------------------------------------------

## Product backlog

Items below are proposed improvements, not features confirmed in the
supplied implementation.

### P0 --- Critical before real-world use

  -----------------------------------------------------------------------
  ID                      Backlog item            User value / acceptance
                                                  criteria
  ----------------------- ----------------------- -----------------------
  US-01                   Improve and validate    Investigate the poor
                          intent classifier       TinyBERT metrics;
                                                  compare models and
                                                  preprocessing; publish
                                                  a held-out evaluation
                                                  before enabling
                                                  automated
                                                  classification

  US-02                   Human review workflow   Store uncertain
                                                  predictions in a review
                                                  queue; allow an
                                                  authorised reviewer to
                                                  confirm/correct the
                                                  intent

  US-03                   Strengthen fraud        Add multilingual,
                          detection               paraphrase, typo, and
                                                  adversarial test
                                                  coverage; make the
                                                  escalation message
                                                  clear and conservative

  US-04                   Protect customer data   Add privacy notice,
                                                  retention rules,
                                                  redaction, secure
                                                  storage, and access
                                                  controls before
                                                  handling real customer
                                                  messages

  US-05                   Add robust error        Handle model loading,
                          handling                Ollama downtime,
                                                  translation failures,
                                                  and invalid outputs
                                                  with actionable
                                                  messages

  US-06                   Security review         Verify that no feature
                                                  asks for or stores
                                                  OTPs, PINs, passwords,
                                                  or full card numbers
  -----------------------------------------------------------------------

### P1 --- Important product improvements

  -----------------------------------------------------------------------
  ID                      Backlog item            User value / acceptance
                                                  criteria
  ----------------------- ----------------------- -----------------------
  US-07                   Persistent prediction   Save records in a
                          history                 database and retrieve
                                                  them across sessions

  US-08                   Search and filter       Filter by date,
                          history                 predicted intent,
                                                  priority, confidence,
                                                  and review status

  US-09                   Reviewer feedback       Let reviewers correct
                          capture                 intent and record
                                                  feedback for future
                                                  model evaluation

  US-10                   Expand FAQ coverage     Build a reviewed FAQ
                                                  catalogue covering more
                                                  Banking77 intents and
                                                  common Indian banking
                                                  workflows

  US-11                   Improve multilingual    Replace phrase-only
                          support                 language detection with
                                                  robust language
                                                  identification and
                                                  validated translation

  US-12                   Add response quality    Check generated answers
                          checks                  for unsupported claims,
                                                  unsafe instructions,
                                                  and language mismatch

  US-13                   Improve sentiment       Evaluate sentiment on
                          handling                Hindi and Marathi or
                                                  clearly label
                                                  translated-English
                                                  sentiment as
                                                  approximate

  US-14                   Add automated test      Add unit and
                          suite                   integration tests for
                                                  preprocessing, fraud
                                                  rules, priority rules,
                                                  model inference, and
                                                  export
  -----------------------------------------------------------------------

### P2 --- Future enhancements

  -----------------------------------------------------------------------
  ID                      Backlog item            User value / acceptance
                                                  criteria
  ----------------------- ----------------------- -----------------------
  US-15                   Agent dashboard         Show volumes, priority
                                                  distribution, common
                                                  intents, and unresolved
                                                  cases

  US-16                   Role-based access       Separate support-agent
                                                  and supervisor
                                                  permissions

  US-17                   Case management         Assign case IDs,
                                                  owners, status, notes,
                                                  and resolution outcomes

  US-18                   Feedback analytics      Track corrections and
                                                  recurring model errors

  US-19                   Model monitoring        Monitor confidence
                                                  distribution, drift,
                                                  latency, and class-wise
                                                  performance

  US-20                   Deployment packaging    Provide Docker
                                                  configuration,
                                                  environment checks, and
                                                  deployment
                                                  documentation

  US-21                   Accessibility and UX    Add clearer error
                                                  states, keyboard
                                                  navigation, and
                                                  responsive layout
                                                  improvements
  -----------------------------------------------------------------------

### Definition of Done

A backlog item is complete when:

-   The acceptance criteria are met.
-   Relevant automated or manual tests pass.
-   Error and edge cases are covered.
-   Documentation is updated.
-   No banking-safety rule is weakened.
-   A reviewer can reproduce the result.

------------------------------------------------------------------------

## Known limitations and risks

1.  **TinyBERT evaluation is poor in the supplied notebook.** Its
    reported accuracy is 15.97% and macro F1 is 8.86%, while the
    TF-IDF + Logistic Regression baseline reports 85.88% accuracy and
    85.81% macro F1. This needs investigation before relying on
    TinyBERT.
2.  **Language detection is heuristic.** The current implementation uses
    word lists and exact phrase mappings; mixed-language or unseen
    phrasing may be misidentified.
3.  **Translation depends on Ollama for unrecognised Hindi/Marathi
    phrases.** If Ollama is unavailable, translation and chatbot
    functionality may fail.
4.  **Sentiment model is English-specific.** Its output after
    translation is not a validated Hindi/Marathi sentiment score.
5.  **Fraud detection is keyword/rule-based.** It can miss novel wording
    and can also flag benign messages. It is not a fraud investigation
    system.
6.  **Priority is rule-based, not a trained risk model.** It should be
    treated as triage assistance, not a final operational decision.
7.  **FAQ coverage is limited.** The supplied FAQ dictionary contains a
    small set of explicit entries and keyword-based fallbacks.
8.  **History is session-only.** Streamlit session state is not a
    durable database and should not be presented as permanent case
    storage.
9.  **No authentication or role-based access is shown.** Do not expose
    the prototype publicly with real customer data.
10. **No live bank integration is shown.** SmartDesk cannot inspect
    accounts, block cards, initiate refunds, verify transactions, or
    file complaints on behalf of customers.

------------------------------------------------------------------------

## Security and responsible use

-   Never request an OTP, ATM PIN, password, CVV, or full card number.
-   Never claim to have accessed a customer's bank account or completed
    a transaction.
-   Never promise a refund, reversal, card unblock, or specific
    resolution time.
-   Direct customers to their bank's official app, website, branch, or
    verified support channel.
-   For suspected financial cyber fraud in India, the predefined
    response points customers to the official cybercrime portal at
    <https://cybercrime.gov.in> and helpline **1930**. Verify current
    official guidance before operational deployment.
-   Treat all model outputs as suggestions requiring human judgement.
-   Do not use this prototype to make final fraud, credit, account, or
    customer eligibility decisions.
-   Do not test with real credentials or real customer
    personal/financial data.
