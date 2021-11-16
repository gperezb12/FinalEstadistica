library(mosaic)
library("factoextra")
library("FactoMineR")
library(ClusterR)
library(cluster)
library(cluster.datasets)
library(tidyverse)
library(gridExtra)
library(dplyr)
library(clue)
library(ggplot2)
library(caret)
library(class)

dioxido1

# Se extraen las columnas numéricas del dataset
data <- dioxido1[2:8]

#Se estandariza la matriz, es decir scaledDioxido = (dioxido1-mean(dioxido1))/sd(dioxido1)
#Cada columna debe tener  media = 0 y desviación estándar = 1
scaledDioxido  <- scale(data)
scaledDioxido 

#Se verifica media =0  para todas las columnas.
colMeans(scaledDioxido)

#Se verifica desviación estándar = 1 para cada columna.
sd(scaledDioxido[,1])

#Analisis de componentes principales

#Matriz de covarianzas
cov(scaledDioxido)

#Análisis de componentes principales según matriz de covarianza, 
#scale = FALSE indica que no se escala la matriz de covarianzas primero.
b <- prcomp(scaledDioxido, center = TRUE, scale = FALSE)
b 
summary(b) 


#Gráficas de componentes principales

#biplot
biplot(b,scale = 1)

#Scree plot
fviz_eig(b, addlabels = TRUE, ylim = c(0, 50))

#plot
plot(b$x[,1],b$x[,2], xlab="PC1 (38.95%)", ylab = "PC2 (21.62%)", main = "PC1 / PC2 - plot")

#correlation circle
fviz_pca_var(b, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE # Avoid text overlapping
)

#Graph of individuals
fviz_pca_ind(b, col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE # Avoid text overlapping (slow if many points)
)

#kmeans y k-clusters

#Sum of squares es para decidir el número de clusters
wssplot <- function(scaledDioxido, nc=15, seed = 20){
  wss <- (nrow(scaledDioxido)-1)*sum(apply(scaledDioxido,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(scaledDioxido, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of groups",
       ylab="Sum of squares within a group")}

wssplot(scaledDioxido, nc = 20)

#Número de clusters
k<-4
clusters <- kmeans(scaledDioxido, centers = k)
dioxido1$clusters <- clusters$cluster


#graficar

plot(dioxido1[c("pob","emp")], 
     col = clusters$cluster, 
     main = "K-means con 2 clusters")

clusters$centers
clusters$centers[c("pob","emp")]

points(clusters$centers[, c("pob","emp")], 
       col = 1:3, pch = 8, cex = 3) 


clusplot(scaledDioxido[, c("pob","emp")],
         clusters$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters"),
         xlab = 'pob',
         ylab = 'emp')

#Clasificación lineal

#Crear la partición de los datos

dioxido <- dioxido1[2:8]
dioxido$oldclusters <- clusters$cluster

inTrain <- createDataPartition(y=dioxido$oldclusters,
                               p=0.8, list=FALSE)
training <- dioxido[inTrain,]
testing <- dioxido[-inTrain,]

dim(training); dim(testing)

#clusters

kMeans1 <- kmeans(training, centers = k)
training$clusters <- as.factor(kMeans1$cluster)

clusplot(training[, c("pob","emp")],
         training$clusters,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters"),
         xlab = 'pob',
         ylab = 'emp')



modFit <- train(clusters ~.,data=subset(training,select=-c(oldclusters)),method="rpart")
table(predict(modFit,training),training$oldclusters)

testClusterPred <- predict(modFit,testing) 
testing$clusters <- testClusterPred
table(testClusterPred ,testing$oldclusters)


clusplot(testing[, c("pob","emp")],
         testing$clusters,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters"),
         xlab = 'pob',
         ylab = 'emp')

#matriz de confusión de training con los datos completos
cm <- table(training$oldclusters, training$clusters)
cm

#matriz de confusión de testing con los datos completos
cm <- table(testing$oldclusters, testing$clusters)
cm

#Detección de outliers con la distancia de Mahalanobis
mahalDist <- mahalanobis(data, colMeans(data), cov(data))
dioxido1$MahalanobisDistance <- round(mahalDist, 1)

dioxido1$outlier <- "No"

#La cota determina desde que distancia son considerados outliers.
cota <- qchisq(p = 0.975 , df = ncol(data))
dioxido1$outlier[dioxido1$MahalanobisDistance > cota] <- "Yes"  

dioxido1$outlier

#Analisis discriminante
data <- dioxido1[2:8]
library(MASS)

clusters <- kmeans(data,3)
ans <- lda(clusters~., data)
ans

#Clasificación con k-vecinos
library(class)
num <- 4
pr <- knn(train,test,cl=train$Cluster, k = num)
test$clusters
test$cluster <- factor(test$clusters)
levels(test$clusters) <- levels(pr)
confusionMatrix(pc,test$clusters)



