from flask import Flask, render_template, request
import pickle
# Load your saved model and vectorizer
vector = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('finalized_model.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("Index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        news = request.form['news']
        print("User Input:", news)

        # ✅ use the variable, not the string "news"
        transformed = vector.transform([news])[0]
        predict = model.predict(transformed)
        print("Prediction:", predict)

        return render_template("prediction.html",
                               prediction_text=f"News headline is → {predict[0]}")
    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=True)
