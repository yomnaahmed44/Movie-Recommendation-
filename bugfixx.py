import sys
import pandas as pd
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QSlider, QComboBox, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz ,process
import random
from PySide6.QtCore import Qt

# Load and clean the dataset
df = pd.read_csv(r"netflix_titles.csv", on_bad_lines='skip')
df['description'] = df['description'].fillna('')

# Define recommendation functions
def fuzzy_match(input_str, choices, limit=3):
    '''Return the best matches using fuzzy logic.'''
    results = process.extract(input_str, choices, limit=limit)
    return [match[0] for match in results]

def content_based_filtering(user_preference, df):
    '''Recommends movies or TV shows based on content similarity.'''
    try:
        matched_descriptions = fuzzy_match(user_preference, df['description'].tolist(), limit=5)
        matched_descriptions = ' '.join(matched_descriptions)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
        user_tfidf = tfidf_vectorizer.transform([matched_descriptions])
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        return cosine_similarities
    except Exception as e:
        print(f"Error in content-based filtering: {e}")
        return []

def fuzzy_score(mood, physical_state, genre):
    '''Calculate a fuzzy score based on mood, physical state, and genre preference.'''
    score = (mood + physical_state) / 2  # Simple average for demonstration purposes
    genre_score = fuzz.token_sort_ratio(genre, 'high-energy')  # Example fuzzy genre match
    return (score + genre_score / 100) / 2  # Normalize the score


def recommend_movie_or_tv_show(user_input, recommended_titles):
    '''Main recommendation function with improved accuracy and diversity.'''
    mood, physical_state, user_preference, genre_input, duration_input, reviews_input, polarity_input, content_type, release_year_input, country_input = user_input

    if not release_year_input:
        release_year_input = str(random.choice(df['release_year'].dropna().unique()))

    filtered_df = df[
        (df['type'].str.lower() == content_type.lower()) &
        (df['listed_in'].str.contains(genre_input, case=False)) &
        (df['release_year'] == int(release_year_input))
    ]

    if filtered_df.empty:
        print("No exact match found, trying to relax constraints for country...")
        filtered_df = df[
            (df['type'].str.lower() == content_type.lower()) &
            (df['listed_in'].str.contains(genre_input, case=False)) &
            (df['release_year'] == int(release_year_input))
        ]

        if filtered_df.empty:
            print("No recommendations after relaxing constraints. Using content-based filtering...")
            cosine_similarities = content_based_filtering(user_preference, df)
            df['similarity_score'] = cosine_similarities
            filtered_df = df.sort_values(by='similarity_score', ascending=False)

            most_similar_item = filtered_df.iloc[0]
            return most_similar_item['title'], most_similar_item['similarity_score']

    cosine_similarities = content_based_filtering(user_preference, filtered_df)
    filtered_df = filtered_df.copy()
    filtered_df['similarity_score'] = cosine_similarities

    filtered_df = filtered_df[~filtered_df['title'].isin(recommended_titles)]

    if not filtered_df.empty:
        filtered_df['fuzzy_score'] = filtered_df.apply(lambda row: fuzzy_score(mood, physical_state, genre_input), axis=1)
        filtered_df['final_score'] = 0.7 * filtered_df['similarity_score'] + 0.3 * filtered_df['fuzzy_score']
        filtered_df = filtered_df.sort_values(by='final_score', ascending=False)

        top_recommendations = filtered_df.head(10)
        top_recommendations = top_recommendations.sample(frac=1).reset_index(drop=True)

        most_similar_item = top_recommendations.iloc[0]
        recommended_titles.append(most_similar_item['title'])

        return most_similar_item['title'], most_similar_item['final_score']
    else:
        return "No new recommendations available, please adjust your criteria.", 0

def add_widget_with_label(layout , labeltext , widget):
    horlay = QHBoxLayout()
    lable = QLabel(labeltext)
    lable.setStyleSheet("font-size : 16px ; font-weight: bold; color : white")
    horlay.addWidget(lable)
    horlay.addWidget(widget)
    layout.addLayout(horlay)




# PySide6 GUI
class MovieRecommendationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Movie and TV Show Recommendation")
        self.setGeometry(100, 100, 400, 500)

        self.recommended_titles = []
        self.init_ui()


    def init_ui(self):
         # Layout setup
        main_layout = QVBoxLayout()


        # Mood Slider
        self.mood_slider = QSlider()
        #self.mood_slider.setOrientation(1)  # Horizontal orientation
        self.mood_slider.setMinimum(0)
        self.mood_slider.setMaximum(10)
        add_widget_with_label(main_layout , 'Mood : ' , self.mood_slider)

        # Physical State Slider
        self.physical_state_slider = QSlider()
        #self.physical_state_slider.setOrientation(0)
        self.physical_state_slider.setMinimum(0)
        self.physical_state_slider.setMaximum(10)
        add_widget_with_label(main_layout , 'Physical State (0-10):' , self.physical_state_slider)

        # User Preference Entry
        self.user_preference_entry = QLineEdit()
        add_widget_with_label(main_layout , 'Preference : ' , self.user_preference_entry)

        # Genre Entry
        self.genre_entry = QLineEdit()
        add_widget_with_label(main_layout , "Genre:" , self.genre_entry)

        # Duration ComboBox
        self.duration_combobox = QComboBox()
        self.duration_combobox.addItems(["Short", "Medium", "Long"])
        #self.duration_combobox.setStyleSheet("background-color : #2E1A47 ; color : white ;")
        add_widget_with_label(main_layout , "Duration :" ,self.duration_combobox )

        # Reviews ComboBox
        self.reviews_combobox = QComboBox()
        self.reviews_combobox.addItems(["Good", "Average", "Bad"])
        #.reviews_combobox.setStyleSheet("background-color : #2E1A47 ; color : white ;")
        add_widget_with_label(main_layout , "Reviews :" , self.reviews_combobox)
        # Polarity ComboBox
        self.polarity_combobox = QComboBox()
        self.polarity_combobox.addItems(["Positive", "Negative", "Neutral"])
        #self.polarity_combobox.setStyleSheet("background-color : #2E1A47 ; color : white ;")
        add_widget_with_label(main_layout , "Polarity :" , self.polarity_combobox)

        # Content Type ComboBox
        self.content_type_combobox = QComboBox()
        self.content_type_combobox.addItems(["Movie", "TV Show"])
        #self.content_type_combobox.setStyleSheet("background-color : #2E1A47 ; color : white ;")
        add_widget_with_label(main_layout , "Content" , self.content_type_combobox)

        # Release Year Entry
        self.release_year_entry = QLineEdit()
        add_widget_with_label(main_layout , "Release Year (optional) :" , self.release_year_entry)

        # Country Entry
        self.country_entry = QLineEdit()
        add_widget_with_label(main_layout , "Country (optional):" , self.country_entry)

        # Submit Button
        self.submit_button = QPushButton("Submit")
        self.submit_button.setFixedSize(200 , 50)
        self.submit_button.clicked.connect(self.submit_inputs)
        main_layout.addWidget(self.submit_button)

        self.setLayout(main_layout)

    def submit_inputs(self):
        try:
            mood_value = self.mood_slider.value()
            physical_state_value = self.physical_state_slider.value()
            user_preference_value = self.user_preference_entry.text().strip().lower()
            genre_input_value = self.genre_entry.text().strip().lower()
            duration_input_value = self.duration_combobox.currentText().strip().lower()
            reviews_input_value = self.reviews_combobox.currentText().strip().lower()
            polarity_input_value = self.polarity_combobox.currentText().strip().lower()
            content_type_value = self.content_type_combobox.currentText().strip().lower()
            release_year_input_value = self.release_year_entry.text().strip()
            country_input_value = self.country_entry.text().strip().title()

            if not country_input_value:
                country_input_value = None

            user_input = (
                mood_value, physical_state_value, user_preference_value,
                genre_input_value, duration_input_value, reviews_input_value,
                polarity_input_value, content_type_value, release_year_input_value, country_input_value
            )

            recommended_title, recommendation_score = recommend_movie_or_tv_show(user_input, self.recommended_titles)
            QMessageBox.information(self, "Recommendation", f"We recommend: {recommended_title} (Score: {recommendation_score})")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

def set_background_color(widget):
    widget.setStyleSheet("""
    QWidget {
        background-color: #141414;  /* Netflix's dark background */
    }

    QLabel, QComboBox, QLineEdit, QSlider, QPushButton {
        color: white; /* White text */
    }

    QPushButton {
        background-color: #E50914; /* Netflix red */
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
    }

    QPushButton:hover {
        background-color: #B20710; /* Darker red on hover */
    }

    QComboBox, QLineEdit {
        background-color: #333333; /* Dark input fields */
        border: 1px solid #E50914; /* Border in Netflix red */
        padding: 5px;
    }

    QSlider::handle {
        background-color: #E50914;  /* Red slider handle */
    }

    QSlider::groove:horizontal {
        background: #333333;
        height: 8px;
        border-radius: 4px;
    }

    QSlider::sub-page:horizontal {
        background: #E50914;
        border-radius: 4px;
    }

    QSlider::add-page:horizontal {
        background: #333333;
        border-radius: 4px;
    }
    """)

class WelcomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome")
        self.setGeometry(100, 100, 400, 500)

        # Welcome Layout
        layout = QVBoxLayout()


        welcome_label = QLabel("BUGFIX!")
        #description_label = QLabel('universe of entertainment')
        #description_label.setStyleSheet('font-size : 16px')
        #description_label.setAlignment(Qt.AlignCenter)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size:25px; font-weight: bold; color: white;")
        layout.addWidget(welcome_label)
        #layout.addWidget(description_label)

        # Start Button
        start_button = QPushButton("Start")
        start_button.setStyleSheet("font-weight:bold;  ")
        start_button.clicked.connect(self.start_clicked)
        #start_button.setFixedWidth(100, 50)
        layout.addWidget(start_button)


        self.setLayout(layout)

        # Set the background color for this window
        set_background_color(self)

    def start_clicked(self):
        # Create the second (main) window and show it
        self.main_window = MovieRecommendationApp()
        set_background_color(self.main_window)
        self.main_window.show()
        # Close the welcome window
        self.close()




# Run the PySide6 application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Create the welcome window
    window = WelcomeWindow()
    # Set the background color for the welcome window
    set_background_color(window)
    # Show the welcome window
    window.show()
    sys.exit(app.exec())














