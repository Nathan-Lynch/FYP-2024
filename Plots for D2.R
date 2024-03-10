# Custom Cartpole v0
library(dplyr)
setwd("C:/Users/natha/Desktop/logfiles/ccp-v0")
getwd()


adaptive_lr = list.files(pattern = "^adaptive_lr_[0-9]+\\.csv")
adap_joined = lapply(adaptive_lr, read.csv)

combined_data = bind_rows(adap_joined)
average_values_adap = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))
#############################################
con_lr = list.files(pattern = "^constant_lr_[0-9]+\\.csv")
con_joined = lapply(con_lr, read.csv)

combined_data = bind_rows(con_joined)
average_values_con = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))
##############################################
lin_lr = list.files(pattern = "^linear_lr_[0-9]+\\.csv")
lin_joined = lapply(lin_lr, read.csv)

combined_data = bind_rows(lin_joined)
average_values_lin = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))
#############################################
exp_lr = list.files(pattern = "^exponential_lr_[0-9]+\\.csv")
exp_joined = lapply(exp_lr, read.csv)

combined_data = bind_rows(exp_joined)
average_values_exp = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))


plot(average_values_con, main="Average Rewards during Training on CustomCartPole-v0",
     xlab = "Timestep", ylab = "Reward", type = "l", col = "blue", lwd = 2, ylim=c(0, 500))

points(average_values_adap, lwd = 2, col = "green", type = "l")
points(average_values_lin, lwd = 2, col = "red", type = "l")
points(average_values_exp, lwd = 2, col = "purple", type = "l")
grid()

legend("bottomright", legend=c("Constant Learning Rate", "Linear Learning Rate",
                              "Exponential Learning Rate", "Adaptive Learning Rate"),
       lty=c(1,1,1,1), col=c("blue", "red", "purple", "green"), lwd = 2)

####################################################################
# CustomCartpole-v1

setwd("C:/Users/natha/Desktop/logfiles/ccp-v1")
getwd()


adaptive_lr = list.files(pattern = "^adaptive_lr_[0-9]+\\.csv")
adap_joined = lapply(adaptive_lr, read.csv)

combined_data = bind_rows(adap_joined)
average_values_adap = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))
#############################################
con_lr = list.files(pattern = "^constant_lr_[0-9]+\\.csv")
con_joined = lapply(con_lr, read.csv)

combined_data = bind_rows(con_joined)
average_values_con = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))
##############################################
lin_lr = list.files(pattern = "^linear_lr_[0-9]+\\.csv")
lin_joined = lapply(lin_lr, read.csv)

combined_data = bind_rows(lin_joined)
average_values_lin = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))
#############################################
exp_lr = list.files(pattern = "^exponential_lr_[0-9]+\\.csv")
exp_joined = lapply(exp_lr, read.csv)

combined_data = bind_rows(exp_joined)
average_values_exp = combined_data %>% group_by(Step) %>% summarise(Average_value = mean(Value))


plot(average_values_con, main="Average Rewards during Training on CustomCartPole-v1",
     xlab = "Timestep", ylab = "Reward", type = "l", col = "blue", lwd = 2, ylim=c(0, 500))

points(average_values_adap, lwd = 2, col = "green", type = "l")
points(average_values_lin, lwd = 2, col = "red", type = "l")
points(average_values_exp, lwd = 2, col = "purple", type = "l")
grid()

legend("bottomright", legend=c("Constant Learning Rate", "Linear Learning Rate",
                               "Exponential Learning Rate", "Adaptive Learning Rate"),
       lty=c(1,1,1,1), col=c("blue", "red", "purple", "green"), lwd = 2)

print(average_values_adap, n=50)
