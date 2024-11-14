from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', title = "Home Page")

# Route TO /contact
@app.route('/contact', method=['GET', 'POST'])
def conatact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Name : {name}, Email : {email}, Message : {message}")
        return redirect(url_for('contact'))
    
    title = "Conatct Page"
    return render_template('contact.html', title = title)

#keep this as is
if __name__ == '__main__':
    app.run(debug=True)