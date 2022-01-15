install.packages("caret")
install.packages("ellipse")
install.packages("kernlab")
install.packages("randomForest")
library(caret)

#Cargar datos
data("iris")
dataset = iris

#Separar datos en 80% training 20% validation
validation_index = createDataPartition(dataset$Species, p=0.80, list=FALSE)
validation = dataset[-validation_index,]
dataset = dataset[validation_index,]

#Ver datos
dim(dataset)
sapply(dataset, class)
head(dataset)
levels(dataset$Species)

#Ver distribucion de datos por nombre de planta
percentaje = prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentaje=percentaje)

#Ver distribucion de atributos
summary(dataset)

#Split inputs and outputs
x = dataset[,1:4]
y = dataset[,5]

# Ver grafica de cada dato (boxplot)
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

#Ver grafica de cada dato (barplot)
plot(y)

# ver graficas (scatterplot matrix)
featurePlot(x=x, y=y, plot="ellipse")

# ver graficas (Box and whislers)
featurePlot(x=x, y=y, plot="box")

#Ver grafica (Density plots) por valor
scales = list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#Usar 10-fold cross validation(https://es.wikipedia.org/wiki/Validaci%C3%B3n_cruzada#Validaci%C3%B3n_cruzada_de_K_iteraciones) para validar el modelo
control = trainControl(method="cv", number=10)
metric = "Accuracy"

#Modelos de entrenamiento, vamos a usar 5 y ver cual es el mejor

#a)Algoritmos lineales
#1)Linear Discriminant Analysis (LDA) https://es.wikipedia.org/wiki/An%C3%A1lisis_discriminante_lineal
set.seed(7)
fit.lda = train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

#b)Algoritmos no lineales
#2)Classification and Regression Trees (CART). https://es.wikipedia.org/wiki/Aprendizaje_basado_en_%C3%A1rboles_de_decisi%C3%B3n
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

#3)k-Nearest Neighbors (kNN). https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_pr%C3%B3ximos
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

#c)Algoritmos avanzados
#4)Support Vector Machines (SVM) with a linear kernel. https://es.wikipedia.org/wiki/M%C3%A1quinas_de_vectores_de_soporte
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

#5)Random Forest (RF) https://es.wikipedia.org/wiki/Random_forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

#Ver presicion de los modelos
resultados = resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(resultados)

#Comparar presicion de los modelos
dotplot(resultados)

#Mejor modelo
print(fit.lda)

#Verificar presicion del modelo usando validation dataset
predictions = predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

Sepal.Length = c(readline(prompt="Longitud de Sépalo: "))
Sepal.Width = c(readline(prompt="Ancho de Sépalo: "))
Petal.Length = c(readline(prompt="Longitud de Pétalo: "))
Petal.Width = c(readline(prompt="Ancho de Pétalo: "))
df = data.frame(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)

predictions2 = predict(fit.lda, df)
print("Tu planta es: ")
print(predictions2)
