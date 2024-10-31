import re
import joblib
import pandas as pd
import PyPDF2
import gradio as gr
from fuzzywuzzy import fuzz

# PATH2 = '/content/drive/MyDrive/ML UJI/Model Testing/'
model_path = '600crf.pkl'

# Function to clean the page
def clean_page(text):
    text = text.replace("Mahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nDirektori Putusan Mahkamah Agung Republik Indonesia\nputusan.mahkamahagung.go.id\n", "")
    text = text.replace("\nDisclaimer\nKepaniteraan Mahkamah Agung Republik Indonesia berusaha untuk selalu mencantumkan informasi paling kini dan akurat sebagai bentuk komitmen Mahkamah Agung untuk pelayanan publik, transparansi dan akuntabilitas\npelaksanaan fungsi peradilan. Namun dalam hal-hal tertentu masih dimungkinkan terjadi permasalahan teknis terkait dengan akurasi dan keterkinian informasi yang kami sajikan, hal mana akan terus kami perbaiki dari waktu kewaktu.\nDalam hal Anda menemukan inakurasi informasi yang termuat pada situs ini atau informasi yang seharusnya ada, namun belum tersedia, maka harap segera hubungi Kepaniteraan Mahkamah Agung RI melalui :\nEmail : kepaniteraan@mahkamahagung.go.id", "")
    text = text.replace("Telp : 021-384 3348 (ext.318)", "")
    text = re.sub(r'\nHalaman \d+ dari \d+ .*', '', text)
    text = re.sub(r'Halaman \d+ dari \d+ .*', '', text)
    text = re.sub(r'\nHal. \d+ dari \d+ .*', '', text)
    text = re.sub(r'Hal. \d+ dari \d+ .*', '', text)
    return text.strip()

# Function to read and clean text from PDF
def read_pdf(file_pdf):
    try:
        pdf_text = ''
        pdf_file = open(file_pdf, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = clean_page(page.extract_text())
            pdf_text += ' ' + text

        pdf_file.close()
        return pdf_text.strip()

    except Exception as e:
        print("Error:", e)

# Function to clean the text
def clean_text(text):
    text = text.replace('P U T U S A N', 'PUTUSAN').replace('T erdakwa', 'Terdakwa').replace('T empat', 'Tempat').replace('T ahun', 'Tahun')
    text = text.replace('P  E  N  E  T  A  P  A  N', 'PENETAPAN').replace('J u m l a h', 'Jumlah').replace('M E N G A D I L I', 'MENGADILI')
    text = re.sub(r'Halaman \d+', ' ', text)
    text = text.replace('\uf0d8', '').replace('\uf0b7', '').replace('\n', ' ')
    text = re.sub(r'([“”"])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function for CRF representation
def representation_crf(text):
    content_results = {"doc": [], "fragment": [], "token": []}

    fragments = text.split(';')

    for fragment_idx, fragment in enumerate(fragments):
        fragment_str = f"fragment:{fragment_idx}"
        doc_str = f"doc:1"

        fragment = re.sub(r'([\/,\.():;])', r' \1 ', fragment)
        tokens = re.findall(r'\S+', fragment)

        for token in tokens:
            content_results["doc"].append(doc_str)
            content_results["fragment"].append(fragment_str)
            content_results["token"].append(token)

    new_data = pd.DataFrame(content_results)

    return new_data

# Fragment getter class
class FragmentGetter(object):

    def __init__(self, data):
        self.n_frag = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(a) for (a) in zip(s['token'].values.tolist())]
        self.grouped = self.data.groupby('fragment').apply(agg_func)
        self.fragments = [f for f in self.grouped]

    def get_next(self):
        try:
            f = self.grouped['Fragment: {}'.format(self.n_frag)]
            self.n_frag += 1
            return f
        except:
            return None

# Function to convert token to features
def token2features(frag, i):
    token = frag[i][0]

    features = {
        'bias': 1.0,
        'token': token
    }

    # Features for previous token
    if i > 0:
        features.update({
            'prev1': frag[i - 1][0]

        })
    else:
        features['BOF'] = True  # Beginning of fragment

    if i > 1:
        features.update({
            'prev2': frag[i - 2][0]
        })

    # Features for next token
    if i < len(frag) - 1:
        features.update({
            'next1': frag[i + 1][0]
        })
    else:
        features['EOF'] = True  # End of fragment

    if i < len(frag) - 2:
        features.update({
            'next2': frag[i + 2][0]
        })

    return features

# Function to convert fragment to features
def frag2features(frag):
    return [token2features(frag, i) for i in range(len(frag))]

def frag2labels(frag):
    return [label for token, label in frag]

# Function to process a single sentence and return the resulting data
def process_sentence(sentence):
    data = representation_crf(sentence)

    # Feature engineering
    X_crf = data.drop(['doc'], axis=1)
    getter = FragmentGetter(X_crf)
    fragments = getter.fragments
    X = [frag2features(f) for f in fragments]

    # Predict
    crf = joblib.load(model_path)
    y_pred = crf.predict(X)

    # Input label to dataset
    flat_predictions = [tag for sentence_tags in y_pred for tag in sentence_tags]
    assert len(flat_predictions) == len(data['token'])
    data['label'] = flat_predictions

    return data

# Function to check entities and collect them
# Function to check entities and collect them
def check_entity(df):
    tokens = df['token'].tolist()
    labels = df['label'].tolist()

    entities = {}
    current_entity = ''
    current_label = ''

    for token, label in zip(tokens, labels):
        if label == 'O':
            if current_entity:
                # Simpan entitas yang selesai jika label saat ini adalah O
                if current_label not in entities:
                    entities[current_label] = set()
                entities[current_label].add(current_entity.strip())
                current_entity = ''
                current_label = ''
            continue

        entity_type = label.split('_')[-1]

        if label.startswith('B_'):
            # Simpan entitas yang selesai jika ada
            if current_entity:
                if current_label not in entities:
                    entities[current_label] = set()
                entities[current_label].add(current_entity.strip())
            
            # Mulai entitas baru
            current_entity = token
            current_label = entity_type

        elif label.startswith('I_') and current_label == entity_type:
            current_entity += ' ' + token

        elif label.startswith('E_') and current_label == entity_type:
            current_entity += ' ' + token
            # Simpan entitas yang lengkap
            if current_label not in entities:
                entities[current_label] = set()
            entities[current_label].add(current_entity.strip())
            current_entity = ''
            current_label = ''

    # Tambahkan entitas terakhir jika ada
    if current_entity:
        if current_label not in entities:
            entities[current_label] = set()
        entities[current_label].add(current_entity.strip())

    return entities



# Function to format the entities for display
def format_entities(entities):
    key_mapping = {
        "VERN": "Nomor Putusan",
        "DEFN": "Terdakwa",
        "CRIA": "Tindak pidana",
        "PENA": "Tuntutan Hukuman",
        "ARTV": "Pasal yang Dilanggar",
        "PUNI": "Putusan Hukuman",
        "JUDP": "Hakim Ketua",
        "JUDG": "Hakim Anggota",
        "TIMV": "Tanggal Perkara",
        "REGI": "Panitera",
        "PROS": "Penuntut Umum"
    }

    def remove_similar(items, threshold=80):
        unique_items = []
        for item in items:
            is_covered = False
            for other_item in unique_items:
                if fuzz.partial_ratio(item, other_item) > threshold:
                    is_covered = True
                    # Keep the longer item if similar
                    if len(item) > len(other_item):
                        unique_items.remove(other_item)
                        unique_items.append(item)
                    break
            if not is_covered:
                unique_items.append(item)
        return unique_items

    for entity in entities:
        entities[entity] = remove_similar(entities[entity])

    formatted_entities = []
    for key, value in entities.items():
        deskripsi = key_mapping.get(key, key)
        formatted_entities.append(f"{deskripsi}: {' | '.join(value)}")
    return '\n'.join(formatted_entities)

# Function to visualize tokens with labels using HighlightedText
# Function to visualize tokens with labels using HighlightedText
def visualize_ner(df):
    tokens = df['token'].tolist()
    labels = df['label'].tolist()

    entities = []
    current_entity = ""
    current_label = ""

    for token, label in zip(tokens, labels):
        if label == "O":
            if current_entity:
                entities.append((current_entity, current_label))
                current_entity = ""
                current_label = ""
            entities.append((token, None))
        else:
            entity_type = label.split('_')[-1]
            if label.startswith('B_'):
                # Simpan entitas yang sedang berlangsung
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = token
                current_label = entity_type
            elif label.startswith('I_') and current_label == entity_type:
                current_entity += " " + token
            elif label.startswith('E_') and current_label == entity_type:
                current_entity += " " + token
                entities.append((current_entity, current_label))
                current_entity = ""
                current_label = ""

    # Tambahkan entitas terakhir jika ada
    if current_entity:
        entities.append((current_entity, current_label))

    return entities


# Function to process the input text
def process_text(input_text):
    pecahan = input_text.split(';')

    results = []
    all_entities = {}
    visualizations = []

    for kalimat in pecahan:
        result = process_sentence(kalimat)
        results.append(result)

        # Extract and collect entities
        entities = check_entity(result)
        for entity_type, entity_set in entities.items():
            if entity_type not in all_entities:
                all_entities[entity_type] = set()
            all_entities[entity_type].update(entity_set)

        # Generate visualization
        visualizations.extend(visualize_ner(result))

    # Combine all results into a single DataFrame
    df = pd.concat(results, ignore_index=True)

    # Convert the entities dictionary to a list of tuples and remove duplicates
    final_entities = {entity_type: list(entity_set) for entity_type, entity_set in all_entities.items()}

    return format_entities(final_entities), visualizations

# Function to handle uploaded PDF
def process_pdf(pdf_file):
    text = read_pdf(pdf_file.name)
    return process_text(clean_text(text))

# Define Gradio interface using Blocks
with gr.Blocks() as iface:
    gr.Markdown("## Entity Extraction")
    gr.Markdown("Enter your text to extract entities or upload a PDF file.")

    with gr.Tab("Input Text"):
        text_input = gr.Textbox(label="Input Text")
        text_button = gr.Button("Extract Entities")
        text_output_entities = gr.Textbox(label="Extracted Entities", lines=10)
        text_output_visualization = gr.HighlightedText(label="Visualization")
        text_button.click(fn=process_text, inputs=text_input, outputs=[text_output_entities, text_output_visualization])

    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="Upload PDF")
        pdf_button = gr.Button("Extract Entities")
        pdf_output_entities = gr.Textbox(label="Extracted Entities", lines=10)
        pdf_output_visualization = gr.HighlightedText(label="Visualization")
        pdf_button.click(fn=process_pdf, inputs=pdf_input, outputs=[pdf_output_entities, pdf_output_visualization])

# Launch Gradio app
iface.launch()
