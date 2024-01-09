# Define the main function
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import io
from PIL import Image
import pickle
import traceback
import re
import nltk
from nltk.corpus import stopwords
col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.write("")

with col2:
    st.image("logo2.png")

with col3:
    st.write("")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"]
     {
       background: linear-gradient( to right, #68abc0,#c6d9e6);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load your preprocessed DataFrame from a Pickle file
with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('df2.pkl', 'rb') as file:
    df2 = pickle.load(file)

# Load your pre-trained content-based recommender function from a Pickle file
def custom_recommender(book_title):
    # ITEM-BASED
    book_title = str(book_title)

    if book_title in df['book_title'].values:
        rating_counts = pd.DataFrame(df['book_title'].value_counts())
        rare_books = rating_counts[rating_counts['book_title'] <= 180].index
        common_books = df[~df['book_title'].isin(rare_books)]

        if book_title in rare_books:
            random = pd.Series(common_books['book_title'].unique()).sample(2).values
            st.write('There are no recommendations for this book')
            st.write('Try:\n')
            st.write('{}'.format(random[0]), '\n')
            st.write('{}'.format(random[1]), '\n')
        else:
            user_book_df = common_books.pivot_table(index=['user_id'],
                                                    columns=['book_title'], values='rating')
            book = user_book_df[book_title]
            recom_data = pd.DataFrame(user_book_df.corrwith(book). \
                                      sort_values(ascending=False)).reset_index(drop=False)

            if book_title in [book for book in recom_data['book_title']]:
                recom_data = recom_data.drop(recom_data[recom_data['book_title'] == book_title].index[0])

            low_rating = []
            for i in recom_data['book_title']:
                if df[df['book_title'] == i]['rating'].mean() < 5:
                    low_rating.append(i)

            if recom_data.shape[0] - len(low_rating) > 5:
                recom_data = recom_data[~recom_data['book_title'].isin(low_rating)]

            recom_data = recom_data[0:1]
            recom_data.columns = ['book_title', 'corr']
            recommended_books = []
            for i in recom_data['book_title']:
                recommended_books.append(i)

            df_new = df[~df['book_title'].isin(recommended_books)]

            # CONTENT-BASED (Title, Author, Publisher, Category)
            rating_counts = pd.DataFrame(df_new['book_title'].value_counts())

            rare_books = rating_counts[rating_counts['book_title'] <= 100].index
            common_books = df_new[~df_new['book_title'].isin(rare_books)]
            common_books = common_books.drop_duplicates(subset=['book_title'])
            common_books.reset_index(inplace=True)
            common_books['index'] = [i for i in range(common_books.shape[0])]
            target_cols = ['book_title', 'book_author', 'publisher', 'Category']
            common_books['combined_features'] = [' '.join(common_books[target_cols].iloc[i,].values) for i in
                                                 range(common_books[target_cols].shape[0])]
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(common_books['combined_features'])
            cosine_sim = cosine_similarity(count_matrix)
            index = common_books[common_books['book_title'] == book_title]['index'].values[0]
            sim_books = list(enumerate(cosine_sim[index]))
            sorted_sim_books = sorted(sim_books, key=lambda x: x[1], reverse=True)[1:2]

            books = []
            for i in range(len(sorted_sim_books)):
                books.append(common_books[common_books['index'] == sorted_sim_books[i][0]]['book_title'].item())

            for i in books:
                recommended_books.append(i)

            df_new = df_new[~df_new['book_title'].isin(recommended_books)]

            # CONTENT-BASED (SUMMARY)
            rating_counts = pd.DataFrame(df_new['book_title'].value_counts())
            rare_books = rating_counts[rating_counts['book_title'] <= 100].index
            common_books = df_new[~df_new['book_title'].isin(rare_books)]

            common_books = common_books.drop_duplicates(subset=['book_title'])
            common_books.reset_index(inplace=True)
            common_books['index'] = [i for i in range(common_books.shape[0])]

            summary_filtered = []
            for i in common_books['Summary']:
                i = re.sub("[^a-zA-Z]", " ", i).lower()
                i = nltk.word_tokenize(i)
                i = [word for word in i if not word in set(stopwords.words("english"))]
                i = " ".join(i)
                summary_filtered.append(i)

            common_books['Summary'] = summary_filtered
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(common_books['Summary'])
            cosine_sim = cosine_similarity(count_matrix)
            index = common_books[common_books['book_title'] == book_title]['index'].values[0]
            sim_books = list(enumerate(cosine_sim[index]))
            sorted_sim_books2 = sorted(sim_books, key=lambda x: x[1], reverse=True)[1:4]
            sorted_sim_books = sorted_sim_books2[:2]
            summary_books = []
            for i in range(len(sorted_sim_books)):
                summary_books.append(
                    common_books[common_books['index'] == sorted_sim_books[i][0]]['book_title'].item())

            for i in summary_books:
                recommended_books.append(i)

            df_new = df_new[~df_new['book_title'].isin(recommended_books)]

            # TOP RATED OF CATEGORY
            category = common_books[common_books['book_title'] == book_title]['Category'].values[0]
            top_rated = common_books[common_books['Category'] == category].groupby('book_title').agg(
                {'rating': 'mean'}).reset_index()

            if top_rated.shape[0] == 1:
                recommended_books.append(
                    common_books[common_books['index'] == sorted_sim_books2[2][0]]['book_title'].item())
            else:
                top_rated.drop(top_rated[top_rated['book_title'] == book_title].index[0], inplace=True)
                top_rated = top_rated.sort_values('rating', ascending=False).iloc[:1]['book_title'].values[0]
                recommended_books.append(top_rated)

            recommended_books.append(book_title)  # Add the original book for reference

            return recommended_books  # Return the list of recommended books

    else:
        st.write("Can't find book in the dataset, please check spelling")


def login():
    st.markdown("<h1 style='text-align: center;'> Login to Readily</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Check if the username and password are correct
    if username == "username" and password == "password":
        st.success("Login successful!")
        return True
    else:
        if st.button("Login"):
            st.error("Invalid username or password. Please try again.")
        return False


# Define a function to convert rating to stars
def rating_to_stars(rating):
    full_stars = int(rating)
    empty_stars = 10 - full_stars
    return f"{'✦' * full_stars}{'✧' * empty_stars}"


# Define the main function
def display_top_rated_books():
    st.title("Our Top Rated Books ")

    # Sort the DataFrame by rating in descending order and get the top 20
    top_rated_books = df.sort_values(by='rating', ascending=False).drop_duplicates(subset='book_title').head(20)

    # Display the top 20 books with their titles, ratings, summaries, and covers
    for i, row in top_rated_books.iterrows():
        book_title = row['book_title']
        rating = row['rating']
        img_l = row['img_l']
        summary = row['Summary']

        # Create a layout with two columns
        col1, col2 = st.columns(2)

        with col1:
            # Display book cover image on the left
            st.image(img_l, caption=f"Book Cover: {book_title}")

        with col2:
            # Display book title, rating, and summary on the right
            st.write(f"Title: {book_title}")
            st.write(f"Rating: {rating_to_stars(rating)}", unsafe_allow_html=True)
            st.write(f"Summary: {summary}")


# Define the main function
def main():
    if login():
        st.markdown("<h1 style='text-align: center'>Readily</h1>", unsafe_allow_html=True)

        # Dropdown for selecting the page
        page = st.selectbox("Select a page:", ["Book Recommendation", "Top Rated Books"])

        if page == "Book Recommendation":
            # Book recommendation page
            book_title = st.selectbox("Select a book title:", df2['books'], key='book_title_dropdown')
            if st.button("Recommend"):
                st.subheader("Recommended Books")
                recommended_books_data = custom_recommender(book_title)
                if recommended_books_data:
                    for recommended_book in recommended_books_data[:5]:
                        # Display recommended books
                        book_info = df[df['book_title'] == recommended_book]
                        img_l = book_info['img_l'].values[0]
                        summary = book_info['Summary'].values[0]
                        rating = book_info['rating'].values[0]

                        # Create a layout with two columns
                        col1, col2 = st.columns(2)

                        with col1:
                            # Display book cover image on the left
                            st.image(img_l, caption=f"Book Cover: {recommended_book}")

                        with col2:
                            # Display book summary in the upper right
                            st.write(f"Summary: {summary}")
                            # Display book rating in star format below the summary
                            st.write(f"Rating: {rating_to_stars(rating)}", unsafe_allow_html=True)

        elif page == "Top Rated Books":
            # Top-rated books page
            display_top_rated_books()


if __name__ == "__main__":
    main()