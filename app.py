import streamlit as st
# import tensorflow_hub as hub
import tensorflow as tf
# from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
import spacy
from spacy.lang.en import English

@st.cache_resource
def model_prediction(abstract):
    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''

    nlp=English()
    sentencizer=nlp.add_pipe("sentencizer")
    doc=nlp(abstract)
    abstract_lines=[str(sent) for sent in list(doc.sents)]

    total_lines_in_sample = len(abstract_lines)
    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
      sample_dict = {}
      sample_dict["text"] = str(line)
      sample_dict["line_number"] = i
      sample_dict["total_lines"] = total_lines_in_sample
      sample_lines.append(sample_dict)

    def split_chars(text):
        return " ".join(list(text))
    
    # Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15) 

    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    model_path = "saved_model.pb"
    loaded_model = tf.keras.models.load_model(model_path)
    test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                   test_abstract_total_lines_one_hot,
                                                   tf.constant(abstract_lines),
                                                   tf.constant(abstract_chars)))
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)

    label_encoder=['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    pred = [label_encoder[i] for i in test_abstract_preds]
    # lines, pred = make_skimlit_predictions(abstract, model, tokenizer, label_encoder)
    # pred, lines = make_predictions(abstract)

    for i, line in enumerate(abstract_lines):
        if pred[i] == 'OBJECTIVE':
            objective = objective + line
        
        elif pred[i] == 'BACKGROUND':
            background = background + line
        
        elif pred[i] == 'METHODS':
            method = method + line
        
        elif pred[i] == 'RESULTS':
            result = result + line
        
        elif pred[i] == 'CONCLUSIONS':
            conclusion = conclusion + line

    return objective, background, method, conclusion, result



def main():
    st.set_page_config(
        page_title="SkimLit",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('SkimLit📄')
    st.caption('### An NLP model to classify abstract sentences into the role they play (e.g. objective, methods, results, etc..) to enable researchers to skim through the literature and dive deeper when necessary.')
    
    # creating model, tokenizer and labelEncoder
    # cnt = 0
    # if cnt == 0:
    #     skimlit_model, tokenizer, label_encoder = create_utils(MODEL_PATH, TOKENIZER_PATH, LABEL_ENOCDER_PATH, EMBEDDING_FILE_PATH)
    #     cnt = 1
    
    col1, col2 = st.columns(2)

    with col1:
        st.write('#### Enter the Abstract Here !!')
        abstract = st.text_area(label='', height=400)
        predict = st.button('Extract !')
    
    # make prediction button logic
    if predict:
        with st.spinner('Wait for prediction....'):
            objective, background, methods, conclusion, result = model_prediction(abstract)
        with col2:
            st.markdown(f'### Objective : ')
            st.write(f'{objective}')
            st.markdown(f'### Background : ')
            st.write(f'{background}')
            st.markdown(f'### Methods : ')
            st.write(f'{methods}')
            st.markdown(f'### Result : ')
            st.write(f'{result}')
            st.markdown(f'### Conclusion : ')
            st.write(f'{conclusion}')



if __name__=='__main__': 
    main()
