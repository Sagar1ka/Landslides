#https://aigeolabs.com/wp-content/uploads/2022/03/Modeling_Landslide_Risk_Data_Centric_Exp_ML.html

rm(list=ls())
# Load the following packages
library(rgdal) # provides an interface to the Geospatial Data Abstraction Library (GDAL)         
library(raster) # provides important functions for importing and handling raster data
library(mapview)  # provides functions to visualize geospatial data
library(sf) # represents simple features as native R objects
library(randomForest) # implements Breiman's RF algorithm
library(dplyr) # provides basic data transformation functions
library(ggplot2) # provides extension of visualizations
library(corrplot) # provides a graphical display of a correlation matrix
library(gridExtra) # provides user-level functions to arrange multiple grid-based plots on a page
library(tmaptools) # provides a set of tools for reading and processing spatial data
library(tmap) # provides a flexible and easy to use approach to create thematic maps
library(plotly) # creates interactive web-based graphs
library(lime)   #  provides local interpretation for ML models 
library(caret) # provides procedures for training and evaluating machine learning models

#  set up the working directory
setwd("C:/Users/s0x999/Desktop/GIS/Landslide/Data")

# Create a list of raster bands that will be used for modeling
Elevation <- brick("Elevation.tif") # ASTER DEM
Slope <- brick("Slope_qgis.tif") # Derived from ASTER DEM
TPI <- brick("TPI_qgis.tif")  # Topographic Position Index (Derived from ASTER DEM)
TRI <- brick("TRI_qgis.tif") # Topographic Ruggedness Index (Derived from ASTER DEM)
Cross_curvature <- brick("Chim_Plan.tif") # Derived from ASTER DEM
Profile_curvature <- brick("Profile_curvature.tif") # Derived from ASTER DEM
Asc_VV <- brick("S1_median_Asc_VV.tif") # Ascending VV
Asc_VH <- brick("S1_median_Asc_VH.tif") # Ascending VV
plot(Elevation)
plot(Slope)
plot(TPI)
plot(TRI)

# Combine or stack the raster layers.
rvars <- stack(Elevation, Slope, TPI, TRI, Cross_curvature, Profile_curvature, Asc_VV, Asc_VH)

# Create an object "ta_data" and import the training data
ta_data <- readOGR(getwd(), "Chimanimani_Landslides_2019_Poly")
plot(ta_data)

# Check the raster variables and training data
print(rvars)
summary(ta_data)

# Create a vector of unique land cover values
uniqueClasses <- unique(ta_data$Class)
uniqueClasses

#Second, we will sample the same number of random points inside all polygons.
# Set the seed
set.seed(27)
for (i in 1:length(uniqueClasses)) {
  # Select only ploygons from ghe current class
  class_data <- subset(ta_data, Class == uniqueClasses[i])
  # Get random points for these polygons
  classpts <- spsample(class_data, type = "random", n = 200)
  # Add class column to the SpatialPoints object
  classpts$class <- rep(uniqueClasses[i], length(classpts))
  if (i == 1) {
    xy <- classpts
  } else {
    xy <- rbind(xy, classpts)
  }
}

#Next, let's plot xy data set (random points) on the Sentinel-1 imagery.
# Set-up timer
timeStart <- proc.time()
# Plot the random points on Sentinel-1 imagery
plotRGB(rvars, r = 7, g = 8, b = 7, stretch = "lin")
#plot(rvars, 4)
points(xy)

proc.time() - timeStart

#Step 2. Prepare training and test data sets. We are going to use the extract() function to extract values from the different raster variables (rvars).
# Set seed
set.seed(27)
# Extract reflectance values from the raster variables and training sample points
train_samples <- extract(rvars, y = xy, cellnumbers = TRUE)
train_samples = data.frame(response = xy$class, train_samples)

# Check the structure of the training data
str(train_samples)


## Check for any duplicates in the training points.
any(duplicated(train_samples$cells))

#if TRUE. Means->The training points contain some duplicates. Therefore, we need to remove the duplicates.
train_samples <- train_samples[!duplicated(train_samples$cells), -2]
names(train_samples)

# Next, let's check the number of the remaining training points using the table() function. 
table(train_samples$response)

#Next, we will split the training data into training and test sets based on the createDataPartition() function from the caret package. The training data will comprise 60%, while the test data set will include 40%.
## Set seed and split training data
set.seed(27)
inTraining <- createDataPartition(train_samples$response, p = .60, list = FALSE)
training <- train_samples[ inTraining,]
testing  <- train_samples[-inTraining,]

# Step 3. Perform exploratory data analysis (EDA)

#Performing exploratory data analysis (EDA) before training machine learning models is very important. 
#We start by checking the descriptive statistics of the training data set using the summary() function.

# Check summary statistics
summary(training) 

# Next, we are going to use the cor() function to check the correlation between predictor variables. Check band correlations
bandCorrelations <- cor(training[, 2:9])
bandCorrelations

# We will display a mixed plot (numbers and plots) to understand the correlation between the predictors., Display  mixed plot
corrplot.mixed(bandCorrelations,lower.col="black", number.cex = .7, upper = "color")

#Step 4. Train and evaluate the random forest (RF) model
#We will specify the random forest (RF) model as shown below.
# Set up the model tuning parameters
fitControl<- trainControl(method = "repeatedcv", number = 10, repeats = 10)

# Set-up timer to check how long it takes to run the RF model
timeStart <- proc.time()

set.seed(27)
# Train the RF model
rf_model <- train(response~.,
                  data = training,
                  #data <- na.omit(training),
                  method = "rf",
                  trControl = fitControl,
                  prox = TRUE,
                  fitBest = FALSE,
                  localImp = TRUE,
                  returnData = TRUE)
proc.time() - timeStart

# Check the RF model performance
print(rf_model)

# check the parameters of the best model.
rf_model$fin

# Next, calculate and display the top predictor variables. Compute RF variable importance
rf_varImp <- varImp(rf_model)
ggplot(rf_varImp) # Display the most important predictor variables

# Prepare a confusion matrix
#pred_rf <- predict(rf_model, newdata = data)
#confusionMatrix(pred_rf, as.factor(data$response))

pred_rf <- predict(rf_model, newdata = testing)
confusionMatrix(pred_rf, as.factor(testing$response))

# Predict land cover
timeStart<- proc.time() # measure computation time
lc_rf <- predict(rvars, rf_model, type = "prob")
proc.time() - timeStart # user time and system time

# Display landslide risk map using the tmap and tmaptools packages
tmap_mode("view")

tm_shape(lc_rf) + tm_raster(style= "quantile", n=7, palette=get_brewer_pal("Reds", n = 7, plot=FALSE)) +
  tm_layout(legend.outside = TRUE)


# Save the landslide risk map in "img" or "geotiff" formats
writeRaster(lc_rf, filename="Chiman_Landslide_Risk.img", datatype='FLT4S', index=1, na.rm=TRUE, progress="window", overwrite=TRUE)

#STEP5 Local Interpretable Model-agnostic Explanations (LIME)

# Remove the response variable (response) 
training_x <- dplyr::select(training, -response)
testing_x <- dplyr::select(testing, -response)

training_y <- dplyr::select(training, response)
testing_y <- dplyr::select(testing, response)

# Create an explainer object
set.seed(27)
explainer_rf <- lime(training_x, rf_model, n_bins = 5, quantile_bins = TRUE)
summary(explainer_rf)

timeStart<- proc.time() # measure computation time
# Explain new observations
explanation_rf <- explain(
  x = testing_x, 
  explainer = explainer_rf, 
  n_permutations = 5000,
  dist_fun = "euclidean",
  kernel_width = .75,
  n_features = 8, 
  feature_select = "highest_weights",
  n_labels = 2
)
proc.time() - timeStart # user time and system time

# Create a plot to visualize the explanaitions
plot_features(explanation_rf[1:50, ], ncol = 2)









