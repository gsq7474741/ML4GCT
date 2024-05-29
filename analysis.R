
install.packages("psych")
install.packages("tidyr")
install.packages("dplyr")
install.packages("broom")
install.packages("purrr")
install.packages("effsize")
install.packages("semPlot")
install.packages("bruceR")
install.packages("lavaan")
library(psych)
library(readxl)
# Expert Cronbach‘s α
#GCT
file_path <- "data/1Expert Rating Score of GCT.xlsx"
data <- read_xlsx(file_path)
alpha_result1 <- alpha(data[, c("Fluency-Rater1", "Fluency-Rater2", "Fluency-Rater3")])
alpha_result2 <- alpha(data[, c("Flexibility-Rater1", "Flexibility-Rater2", "Flexibility-Rater3")])
alpha_result3 <- alpha(data[, c("Originality-Rater1", "Originality-Rater2", "Originality-Rater3")])

print(alpha_result1$total$std.alpha)   
print(alpha_result2$total$std.alpha) 
print(alpha_result3$total$std.alpha)   

#Picture Completion
file_path <- "data/2Expert Rating Score of Criterion Score.xlsx"
data <- read_xlsx(file_path)
alpha_result4 <- alpha(data[, c("PCrater1Total", "PCrater2Total", "PCrater3Total")])
alpha_result5 <- alpha(data[, c("PCrater1Flu", "PCrater2Flu", "PCrater3Flu")])
alpha_result6 <- alpha(data[, c("PCrater1Orig", "PCrater2Orig", "PCrater3Orig")])

 
print(alpha_result4$total$std.alpha)   
print(alpha_result5$total$std.alpha) 
print(alpha_result6$total$std.alpha)  



#Improve Toy Task
file_path <- "data/2Expert Rating Score of Criterion Score.xlsx"
data <- read_xlsx(file_path)

alpha_result7 <- alpha(data[, c("ITrater1Total", "ITrater2Total", "ITrater3Total")])
alpha_result8 <- alpha(data[, c("ITrater1Flu", "ITrater2Flu", "ITrater3Flu")])
alpha_result9 <- alpha(data[, c("ITrater1Flex", "ITrater2Flex", "ITrater3Flex")])
alpha_result10 <- alpha(data[, c("ITrater1Orig", "ITrater2Orig", "ITrater3Orig")])


print(alpha_result7$total$std.alpha)   
print(alpha_result8$total$std.alpha) 
print(alpha_result9$total$std.alpha)  
print(alpha_result10$total$std.alpha)   


#Discriminant Validity
  #Chi-square Goodness of Fit Test
library(tidyr) 
library(dplyr) 
library(readxl)
library(broom)
library(purrr)
library(effsize)
file_path <- "data/3Discriminant Validity.xlsx"

data <- read_xlsx(file_path)
chi_square_results <- list()
for (var in c("A2", "A3", "A4", "B1", "B2", "B3", "C2")) {
  cont_table_var <- table(data$ScoreGroup, data[[var]])
  chi_square_result_var <- chisq.test(cont_table_var)
  chi_square_results[[var]] <- chi_square_result_var
  {
    cat("Variable:", var, "\n")
    print(chi_square_results[[var]])
    cat("\n\n")
  }
}
 
#Independent samples T test
library(readxl)
library(dplyr)
library(purrr)
library(broom)
library(effsize)
file_path <- "data/3Discriminant Validity.xlsx"

data <- read_xlsx(file_path)

variables <- c("A1", "B4", "B5", "C1", "C3")

t_test_results <- purrr::map_dfr(variables, ~ {
  t_test <- t.test(reformulate("ScoreGroup", response = .x), data = data, var.equal=FALSE)
  
  cohen_d <- cohen.d(data[[.x]], data$ScoreGroup, var.equal=FALSE)$estimate[[1]]
  
  tidy_t_test <- tibble(
    Variable = .x,
    df = t_test$parameter,  
    t_value = t_test$statistic[1],  
    p_value = t_test$p.value,  
    Cohen_d = cohen_d  
  )
  tidy_t_test})  

print(t_test_results)


#Construct Validity——CFA
library("readxl")
library("bruceR")
library("lavaan")
#11Features
read_xlsx("data/4CFA.xlsx")
file_path <- "data/4CFA.xlsx"
data <- read_xlsx(file_path)
model <- '
  # Factor
    f1 =~ A1 + A2 + A3 + A4
    f2 =~ B1 + B2 + B3 + B4 + B5
    f3 =~ C1 + C2
  # Measurement error
    A1 ~~ A2
    A1 ~~ A3
    A1 ~~ A4
    A2 ~~ A3
    A2 ~~ A4
    A3 ~~ A4
    B1 ~~ B2
    B1 ~~ B3
    B1 ~~ B4
    B1 ~~ B5
    B2 ~~ B3
    B2 ~~ B4
    B2 ~~ B5
    B3 ~~ B4
    B3 ~~ B5
    B4 ~~ B5
    C1 ~~ C2'
fit <- cfa(model, data = data, estimator = "DWLS")
summary(fit, fit.measures = TRUE)


library("semPlot")
library("lavaan")
threefactor.model <- "A =~ A1+A2+A3+A4;B =~ B1+B2+B3+B4+B5; C =~ C1+C2"
threefactor.fit<- cfa(model = threefactor.model, data=df, estimator = "DWLS")
semPaths(object = threefactor.fit,
         whatLabels = "std",
         layout = "spring",label.cex=1.5,
         edge.label.cex=0.5, font=2,  
         edge.color="black", fixedStyle = c("black",1),freeStyle = c("black",1),esize =1,
         rotation=1)

#10Features
read_xlsx("data/4CFA.xlsx")
file_path <- "data/4CFA.xlsx"
data <- read_xlsx(file_path)
model <- '
  # Factor
    f1 =~ A1 + A2 + A3 + A4
    f2 =~ B1 + B2 + B3 + B4
    f3 =~ C1 + C2
  # Measurement error
    A1 ~~ A2
    A1 ~~ A3
    A1 ~~ A4
    A2 ~~ A3
    A2 ~~ A4
    A3 ~~ A4
    B1 ~~ B2
    B1 ~~ B3
    B1 ~~ B4
    B2 ~~ B3
    B2 ~~ B4
    B3 ~~ B4
    C1 ~~ C2'
fit <- cfa(model, data = data, estimator = "DWLS")
summary(fit, fit.measures = TRUE)


library("semPlot")
library("lavaan")
threefactor.model <- "A =~ A1+A2+A3+A4;B =~ B1+B2+B3+B4; C =~ C1+C2"
threefactor.fit<- cfa(model = threefactor.model, data=df, estimator = "DWLS")
semPaths(object = threefactor.fit,
         whatLabels = "std",
         layout = "spring",label.cex=1.5,
         edge.label.cex=0.5, font=2,  
         edge.color="black", fixedStyle = c("black",1),freeStyle = c("black",1),esize = 1,
         rotation=1)

#Criterion Validity
file_path <- "data/5Data Used for Criterion Validity.xlsx"
data <- read_xlsx(file_path)
data[, c("PC", "IT", "GCT-T", "GCT-Flu", "GCT-O", "GCT-Fle")] <- sapply(data[, c("PC", "IT", "GCT-T", "GCT-Flu", "GCT-O", "GCT-Fle")], as.numeric)
correlation_matrix <- cor(data[, c("PC", "IT", "GCT-T", "GCT-Flu", "GCT-O", "GCT-Fle")])
print(correlation_matrix)

         