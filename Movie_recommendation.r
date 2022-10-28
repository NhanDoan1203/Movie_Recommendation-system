#importing library
library(recommenderlab)
library(ggplot2)
library(data.table)
library(reshape2)

# Retrieve and display data
setwd('d:\\Movie_recommendation\\IMDB-Dataset')
movie_data <- read.csv("movies.csv", stringsAsFactors=FALSE)
rating_data <- read.csv("ratings.csv")

# structure the data
str(movie_data)
str(rating_data)

data.table(movie_data)
data.table(rating_data)

# Summary data
summary(movie_data)
summary(rating_data)
head(movie_data)
head(rating_data)

# Creating a one-hot encoding to create a matrix that comprises of corresponding genres for each of the films.
movie_genre <- as.data.frame(movie_data$genres, stringsAsFactors = FALSE)
library(data.table)
movie_genre1 <- as.data.frame(tstrsplit(movie_genre[,1], '[|]',type.convert=TRUE),stringsAsFactors = FALSE) 

colnames(movie_genre1) <-c(1:10)
list_genre <- c("Action", "Adventure", "Animation", "Children", "Comedy", 
                "Crime","Documentary", "Drama", "Fantasy","Film-Noir", "Horror", 
                "Musical", "Mystery","Romance","Sci-Fi", "Thriller", "War", "Western")
genre_mat1 <- matrix(0,10330,18)
genre_mat1[1,]<- list_genre
colnames(genre_mat1)<- list_genre
for (index in 1:nrow(movie_genre1)){
  for (col in 1:ncol(movie_genre1)){
    gen_col = which(genre_mat1[1,] == movie_genre1[index,col])
    genre_mat1[index + 1, gen_col] <-1
  }
}
genre_mat2 <- as.data.frame(genre_mat1[-1,], stringsAsFactors = FALSE)
for (col in 1:ncol(genre_mat2)){
  genre_mat2[,col] <- as.integer(genre_mat2[,col])
}
str(genre_mat2)

# Creating search engine
SearchMat <- cbind(movie_data[,1:2], genre_mat2[])
head(SearchMat)

# Sparse matrix\
ratingMatrix <- dcast(rating_data, userId~movieId, value.var = "rating", na.rm=FALSE)
ratingMatrix <- as.matrix(ratingMatrix[,-1])
ratingMatrix <- as(ratingMatrix, "realRatingMatrix")
ratingMatrix

# Overview some important parameters for building recommendation systems for movies
recommendation_model <- recommenderRegistry$get_entries(dataType = "realRatingMatrix")
names(recommendation_model)
lapply(recommendation_model, "[[", "description")

# Collaborative Filtering
recommendation_model$IBCF_realRatingMatrix$parameters

# Compute similarities between users
similarity_mat <- similarity(ratingMatrix[1:4,], method = "cosine", which = "users")
as.matrix(similarity_mat)
image(as.matrix(similarity_mat), main = "User's Similarity")

# Compute similarities between movies
movie_similarity <- similarity(ratingMatrix[1:4,], method = "cosine", which = "users")
as.matrix(similarity_mat)
image(as.matrix(similarity_mat), main = "Movie Similarity")

# Unique rating value 
rating_value <- as.vector(ratingMatrix@data)
unique(rating_value)

# number of movie ratings
Rating_table <- table(rating_value)
Rating_table

# Visualize most viewed movies
library(ggplot2)
view_of_movie <- colCounts(ratingMatrix) #count number of views
table_of_view <- data.frame(movies = names(view_of_movie), views = view_of_movie) # Create dataframe of views
table_of_view <- table_of_view[order(table_of_view$views, decreasing = TRUE),] # sort data by number of views
table_of_view$title <- NA

for (index in 1:10325){
  table_of_view[index,3] <- as.character(subset(movie_data,movie_data$movieId == table_of_view[index,1])$title)
}
table_of_view[1:6,]

# Plotting the data
ggplot(table_of_view[1:6, ], aes(x = title, y = views)) +
  geom_bar(stat="identity", fill = 'orange') +
  geom_text(aes(label=views), vjust=-0.3, size= 3.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("TOTAL VIEWS FOR MOST WATCHED MOVIES")

# Heatmap of rating matrix

image(ratingMatrix[1:30,1:35], axes = FALSE, main = "30 x 35 HEATMAP")

# Data preparation
movie_ratings <- ratingMatrix[rowCounts(ratingMatrix)>50, colCounts(ratingMatrix)>50]

# Heat map for top user and movies
minimum_movies <- quantile(rowCounts(movie_ratings), 0.98)
minimum_users <- quantile(colCounts(movie_ratings), 0.98)
image(movie_ratings[rowCounts(movie_ratings) > minimum_movies, 
                    colCounts(movie_ratings) > minimum_users],
      main = "HEAT MAP FOR TOP MOVIES AND USERS")


# Average rating per user "VISUALIZTION"
average_ratings <- rowMeans(movie_ratings)
qplot(average_ratings, fill=I("steelblue"), col=I("black"), bins = 30) + ggtitle("AVERAGE RATING PER USER")

# Normalizing the data
normalized_ratings <- normalize(movie_ratings)
sum(rowMeans(normalized_ratings)> 0.00001)

# Heatmap for normalized value
image(normalized_ratings[rowCounts(normalized_ratings) > minimum_movies, 
                         colCounts(normalized_ratings) > minimum_users], 
      main = "HEATMAP OF THE NORMALIZATION OF THE TOP USERS")

# Data Binarization
binary_minimum_movies <- quantile(rowCounts(movie_ratings), 0.93)
binary_minimum_users <- quantile(colCounts(movie_ratings), 0.93)

top_rated_movies <- binarize(movie_ratings, minRating = 3.5)
image(top_rated_movies[rowCounts(movie_ratings) > binary_minimum_movies,
                       colCounts(movie_ratings) > binary_minimum_users],
      main = "HEATMAP OF TOP USERS AND MOVIES")

# Collaborative filtering system
sampled_data <- sample(x = c(TRUE, FALSE),
                      size = nrow(movie_ratings),
                      replace = TRUE,
                      prob = c(0.8, 0.2))
training_data <- movie_ratings[sampled_data, ]
testing_data <- movie_ratings[!sampled_data, ]

# Creating recommendation system
rec_system <- recommenderRegistry$get_entries(dataType ="realRatingMatrix")
rec_system$IBCF_realRatingMatrix$parameters
rec_model <- Recommender(data = training_data,
                              method = "IBCF",
                              parameter = list(k = 30))
rec_model

class(rec_model)

# Data science recommendation model
information <- getModel(rec_model)
class(information$sim)
dim(information$sim)
top_items <- 20
image(information$sim[1:top_items, 1:top_items],
      main = "HEATMAP OF THE MODEL")

# Visualize sum of rows and columns with the similarity of the objects above 0
sum_rows <- rowSums(model_info$sim > 0)
table(sum_rows)
sum_cols <- colSums(model_info$sim > 0)
qplot(sum_cols, fill=I("steelblue"), col=I("red"))+ ggtitle("DISTRIBUTION OF THE COLUMN COUNT")

# Number of recommendation to users
top_recommendations <- 15
prediction_module <- predict(object = reco_model,
                                     newdata = testing_data,
                                     n = top_recommendations)
prediction_module

# Recommendation for the first user
user <- prediction_module@items[[1]] 
movies_user <- prediction_module@itemLabels[user1]
movies_user1 <- movies_user
for (index in 1:10){
  movies_user1[index] <- as.character(subset(movie_data,
                                             movie_data$movieId == movies_user[index])$title)
}
movies_user2
# matrix with the recommendations for each user
rec_matrix <- sapply(prediction_module@items,
                                function(x){ as.integer(colnames(movie_ratings)[x]) }) 
rec_matrix[,1:4]
# Distribution of the Number of Items for IBCF
number_of_items <- factor(table(recommendation_matrix))
chart_title <- "Distribution of the Number of Items for IBCF"
qplot(number_of_items, fill=I("steelblue"), col=I("red")) + ggtitle(chart_title)

number_of_items_sorted <- sort(number_of_items, decreasing = TRUE)
number_of_items_top <- head(number_of_items_sorted, n = 4)
table_top <- data.frame(as.integer(names(number_of_items_top)),


