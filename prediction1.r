rm( list = ls() )

require( caret )
#require( ggplot2 )
#require( C50 )

numMiss <- function( df ) {
# a function to return information on the missing values
# in a data frame  
  nm <- vector( mode = "numeric", length = ncol( df ) )
  nm.perc <- vector( mode = "numeric", length = ncol( df ) )
  for( i in ( 1 : ncol( df ) ) ) {
    var <- names( df )[ i ]
    nm[ i ] <- sum( as.numeric( is.na( df[ , i ] ) ) )
    nm.perc[ i ] <- round( nm[ i ] / nrow( df ), 3 )
  }
  return( list( vars = names( df ), nm = nm, nm.perc = nm.perc ) )
}

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

setwd("~/GitHub/Machine_Learning")

# watch for "#DIV/0!"
df <- read.table ( file = "pml-training.csv", header = TRUE, 
      sep = ",", quote = "\"'", stringsAsFactors = FALSE,
      dec = ".", row.names = NULL, 
      na.strings = c( "NA", "#DIV/0!" ) )

df[ , 1 ] <- NULL

table( df$classe )

# find variables with too many missing values and delete them
delMis <- numMiss( df )$nm.perc > 0.9
sum( delMis )

df <- df[ , -which( delMis ) ]

# correlations between numeric predictors
nums <- sapply( df, is.numeric )
corMat <- cor( df[ , nums ] )

# remove correlated variables
delCor <- findCorrelation(corMat, cutoff = .90, verbose = FALSE )
delCor
df <- df[ , -delCor ]


# re-arrange some columns: two character variables before class
df <- df[ , c( 2:3, 5:51, 1, 4, 52 )]

set.seed( 1234 )

inT <- createDataPartition( df$classe, p = 0.6,
                           list = FALSE )

training <- df[ inT, ]
testing <- df[ -inT, ]

rm( df )

training <- within( training, {
  user_name <- as.factor( user_name  )
  new_window <- as.factor( new_window  )
  classe <- as.factor( classe  )
  }
)

testing <- within( testing, {
  user_name <- as.factor( user_name  )
  new_window <- as.factor( new_window  )
  classe <- as.factor( classe  )
  }
)



xTrans <- preProcess( training[ , 1:49 ], method = c("center", "scale") )
tmp <- predict( xTrans, training[ , 1:49 ] )
training <- cbind( tmp, training[ , 50:52 ] )

tmp <- predict( xTrans, testing[ , 1:49 ] )
testing <- cbind( tmp, testing[ , 50:52 ] )

row.names( training )<- 1:nrow( training )
row.names( testing )<- 1:nrow( testing )

options( warn = -1 )

model1 <- train( classe  ~., method = "C5.0Tree", data = training, 
control = C5.0Control( winnow = TRUE ) )

options( warn = 0 )

#model1 <- train( classe  ~., method = "C5.0Tree", data = training, 
#trControl = trainControl( method = "boot632", number = 10 ) )

save( model1, file = "model1.Rdata" )
load( file = "model1.Rdata" )

tab1 <- predict( model1, newdata = training )
confusionMatrix( tab1, training$classe, dnn = c( "Predicted", "Actual" ) )

tab2 <- predict( model1, newdata = testing )
confusionMatrix( tab2, testing$classe, dnn = c( "Predicted", "Actual" ) )

v <- varImp( model1 )
v
dotPlot(  varImp( model1 ), top = 20 )

testing <- within( testing, {
  treeVar <- ifelse( roll_belt > 1.043879 & pitch_arm > 1.634051, 'Node 1',
             ifelse( roll_belt > 1.043879 & pitch_arm <= 1.634051, 'Node 2',
             ifelse( roll_belt <= 1.043879 & pitch_forearm > -1.623873, 'Node 3', 'Node 4' )))
  }
)          

ggplot( data = testing, aes( x = treeVar, fill =  classe) ) + 
    geom_bar() + scale_fill_brewer( palette = 1 ) +
    labs( title = "Top nodes in the tree - testing set", 
          y = "Count", x = "Node", fill = "Class") 		

summary( model1$finalModel )

# predicting classes of new cases

df.new <- read.table ( file = "pml-testing.csv", header = TRUE, 
      sep = ",", quote = "\"'", stringsAsFactors = FALSE,
      dec = ".", row.names = NULL, 
      na.strings = c( "NA", "#DIV/0!" ) )

df.new[ , 1 ] <- NULL

# leave same predictors in the same order
common_names <- intersect( names( training ), names( df.new ) )
df.new <- df.new[ , common_names ]

# center and scale using the transformation in the training set
tmp <- predict( xTrans, df.new[ , 1:49 ] )
df.new <- cbind( tmp, df.new[ , 50:51 ] )

tab3 <- predict( model1, newdata = df.new )

pml_write_files( tab3 )