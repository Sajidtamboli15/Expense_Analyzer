from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pickle

app = Flask(__name__)
app.secret_key = "secretkey"   # ⚠️ change this in production

# Configure DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:sajid1508@localhost/expense_tracker'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Expense Model
class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    expense_date = db.Column(db.Date, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_with_threshold(text, threshold=0.3):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    max_prob = max(probs)
    if max_prob < threshold:
        return "Other"
    return model.classes_[probs.argmax()]

# Signup route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]   # ⚠️ hash in real apps
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("⚠️ Username already taken. Please choose another.", "warning")
            return redirect("/signup")
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash("✅ Account created successfully! Please login.", "success")
        return redirect("/login")
    return render_template("signup.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            flash("✅ Logged in successfully!", "success")
            return redirect("/")
        else:
            flash("❌ Invalid username or password. Please try again.", "danger")
    return render_template("login.html")

# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("✅ You have been logged out.", "info")
    return redirect("/login")

# Home route → show one month at a time
@app.route("/")
@login_required
def index():
    expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.expense_date).all()
    if not expenses:
        return render_template("index.html")

    df = pd.DataFrame([{
        "id": e.id,
        "item": e.item,
        "category": e.category,
        "amount": float(e.amount),
        "date": e.expense_date
    } for e in expenses])

    df["Month"] = df["date"].apply(lambda d: d.strftime("%B %Y"))
    months = sorted(df["Month"].unique())

    selected_month = request.args.get("month")
    if not selected_month:
        selected_month = months[-1]

    df_month = df[df["Month"] == selected_month]

    summary = df_month.groupby("category")["amount"].sum().to_dict()
    grand_total = sum(summary.values())

    categories = df_month["category"].unique()
    palette = px.colors.qualitative.Dark2
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    fig_pie = px.pie(df_month.groupby("category")["amount"].sum().reset_index(),
                     values="amount", names="category",
                     title=f"{selected_month} - Expense Distribution",
                     color="category", color_discrete_map=color_map)
    pie_html = pio.to_html(fig_pie, full_html=False)

    fig_bar = px.bar(df_month.groupby("category")["amount"].sum().reset_index(),
                     x="category", y="amount",
                     title=f"{selected_month} - Total Expenses per Category",
                     text_auto=True, color="category",
                     color_discrete_map=color_map)
    bar_html = pio.to_html(fig_bar, full_html=False)

    rows = df_month.to_dict("records")

    return render_template("index.html", months=months, selected_month=selected_month,
                           rows=rows, summary=summary, grand_total=grand_total,
                           pie_chart=pie_html, bar_chart=bar_html)

# Manual entry route
@app.route("/add_expense", methods=["POST"])
@login_required
def add_expense():
    item = request.form["item"]
    amount = float(request.form["amount"])
    expense_date = datetime.strptime(request.form["expense_date"], "%Y-%m-%d").date()
    category = predict_with_threshold(item)

    expense = Expense(item=item, category=category, amount=amount,
                      expense_date=expense_date, user_id=current_user.id)
    db.session.add(expense)
    db.session.commit()
    flash("✅ Expense added successfully!", "success")
    return redirect("/")

# Delete route
@app.route("/delete/<int:expense_id>")
@login_required
def delete_expense(expense_id):
    expense = Expense.query.get_or_404(expense_id)
    if expense.user_id != current_user.id:
        flash("❌ Unauthorized action.", "danger")
        return redirect("/")
    db.session.delete(expense)
    db.session.commit()
    flash("✅ Expense deleted successfully!", "info")
    return redirect("/")

# CSV upload route
@app.route("/csv", methods=["GET", "POST"])
@login_required
def csv_page():
    if request.method == "POST":
        file = request.files["file"]
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]

        if "description" not in df.columns or "amount" not in df.columns:
            flash("❌ CSV must have 'Description' and 'Amount' columns.", "danger")
            return render_template("csv.html")

        df["Predicted_Category"] = df["description"].apply(lambda x: predict_with_threshold(str(x)))
        df = df[["description", "amount", "Predicted_Category"]]
        rows = df.to_dict("records")

        summary = df.groupby("Predicted_Category")["amount"].sum().to_dict()
        grand_total = sum(summary.values())

        categories = df["Predicted_Category"].unique()
        palette = px.colors.qualitative.Dark2
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

        fig_pie = px.pie(df.groupby("Predicted_Category")["amount"].sum().reset_index(),
                         values="amount", names="Predicted_Category",
                         title="Expense Distribution by Category",
                         color="Predicted_Category", color_discrete_map=color_map)
        pie_html = pio.to_html(fig_pie, full_html=False)

        fig_bar = px.bar(df.groupby("Predicted_Category")["amount"].sum().reset_index(),
                         x="Predicted_Category", y="amount",
                         title="Total Expenses per Category",
                         text_auto=True, color="Predicted_Category",
                         color_discrete_map=color_map)
        bar_html = pio.to_html(fig_bar, full_html=False)

        return render_template("csv.html", rows=rows, summary=summary, grand_total=grand_total,
                               pie_chart=pie_html, bar_chart=bar_html)

    return render_template("csv.html")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
