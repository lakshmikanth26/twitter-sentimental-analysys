from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        tweet_text = request.form['tweet_text']

        # Perform sentiment analysis
        result = sentiment_analyzer(tweet_text)

        # Extract sentiment label and score
        labels = ['Negative', 'Neutral', 'Positive']
        sentiment_label = result[0]['label']
        sentiment_score = result[0]['score']
        label_mapping = {'LABEL_0': labels[0], 'LABEL_1': labels[1], 'LABEL_2': labels[2]}
        sentiment = label_mapping.get(sentiment_label, 'Unknown')
        return render_template('index.html', tweet_text=tweet_text, sentiment_label=sentiment, sentiment_score=sentiment_score)

if __name__ == '__main__':
    app.run(debug=True)
