
# This code is used to estimate an association between pre-existing comorbidity and physical impairment at hospital discharge among patients with sepsis


#----------------------------------
# packages
#----------------------------------

library(tidyverse)
library(magrittr)
library(GGally)
library(finalfit)
library(Hmisc)
library(survival)
library(survminer)
library(ggpubr)
library(mice)
library(egg)
library(lmtest)
library(logbin)



#----------------------------------
# variables in use for imputation
#----------------------------------

numerical = c(
    # baseline 
    "age_ICUadmission", "hp_days_log", "life_space_befo_ad_to_hp2",
    # conditions
    "APACHE2_ICU_admission", "SOFA_ICUadmission",
    "lactate_max", "EQ5D_vas_initial", "EQ5D_vas_last",
    "total_FIM_score_initial",  "total_FIM_score_last"
    )
    
categorical = c(
    # baseline
    "sex2", 
    # exposure variables
    "comor",  
    "cardiopul", "cerebro", "metabolic", "cancer",
    # conditions
    "dnar_atleast",  "medical", "shock", "ARDS_bi", "AKI2",
    "blood_culture", "septic.encephalopathy",  "delirium",
    "septic.cardiomyopathy",
    # type_of_infection
    "type_of_infection_community_acquired",
    "type_of_infection_healthcare_associated",
    "type_of_infection_nosocomial",
    # admission route
    "ED_cli", "elective", "inpatients", "transf",
    # site of infection
    "respi2", "abdominal2", "uri2", "others2",  "unknown2",
    # intervention
    "RRT2", "MV",  "NIV",  "NHF",
    "propofol", "midazolam", "dex",
    "noradrenaline2", "hydrocortisone2", "vasopressin2",
    "CVC",
    "rehab_hp", "rehab_did_ICU",
    # discharge disposition
    "acute_care", "home", "home_health_services", "long_term", "others")

bias = c('previous_dependence',"mortality_3day2", "hp_death_after_3days")
outcome = c('dependent_hp_discharge')





#----------------------------------
# univariate analysis
#----------------------------------
# study population
pop = 
    data %>% 
    filter(is.na(previous_dependence)==T| previous_dependence==0) %>%
    filter(mortality_3day2==0) %>%
    filter(hp_death_after_3days == 0)

# imputation dataset
# exclude bias factors from available variables
impu_base_nu   = numerical
impu_base_cate = categorical
explanatory = c(impu_base_nu, impu_base_cate)
# bias terms were removed because even though this is imputation purpose, study population usually did not include these terms.
dependent = c("dependent_hp_discharge")


# log binomial
# prepare imputation set
cols = c(explanatory, dependent)
temp = pop %>% dplyr::select(all_of(cols))
imp_base = mice(temp, maxit = 5, m = 10, seed = 25)
imp_tab_base = complete(imp_base, 'long') # extract imputed dataset that will be used for inital values for weights

# function for estimation
test_logbino = function(imp_mice, mice_table, glm_formula, n_covariate){
        fit = with(imp_mice, glm(formula(glm_formula), family = binomial(link='log'),
                        start=c(log(mean(mice_table$dependent_hp_discharge)), rep(0, n_covariate+1))))  
        sum_ = summary(pool(fit), conf.int=T, exponentiate=T) 
        res = sum_[2,]
        return(res) 
}


# Each comorbidity using the final model
comor_category  = c('comor','cardiopul','cerebro', 'metabolic', 'cancer')
res = list()
for (i in comor_category){
    glm_formula = paste0('dependent_hp_discharge ~ ', i)  
    res[[i]] = test_logbino(imp_base, imp_tab_base, glm_formula, n_covariate=0)
}    
saveRDS(res, './results/res_univariate.rds')    
    




#----------------------------------
# Adjusting only covariates
#----------------------------------

# study population
pop = 
    data %>% 
    filter(is.na(previous_dependence)==T| previous_dependence==0) %>%
    filter(mortality_3day2==0) %>%
    filter(hp_death_after_3days == 0)


# imputation dataset
# exclude bias factors from available variables
impu_base_nu   = numerical
impu_base_cate = categorical
explanatory = c(impu_base_nu, impu_base_cate)
# bias terms were removed because even though this is imputation purpose, study population usually did not include these terms.
dependent = c("dependent_hp_discharge")


# log binomial
# prepare imputation set
cols = c(explanatory, dependent)
temp = pop %>% dplyr::select(all_of(cols))
imp_base = mice(temp, maxit = 5, m = 10, seed = 25)
imp_tab_base = complete(imp_base, 'long') # used for inital values for weights

test_logbino = function(imp_mice, mice_table, glm_formula, n_covariate){
        fit = with(imp_mice, glm(formula(glm_formula), family = binomial(link='log'),
                        start=c(log(mean(mice_table$dependent_hp_discharge)), rep(0, n_covariate+1))))  
        sum_ = summary(pool(fit), conf.int=T, exponentiate=T) 
        res = sum_[2,]
        return(res) 
}

res = list()

glm_formula = 'dependent_hp_discharge ~ comor + 
                    age_ICUadmission + 
                    sex2+
                    SOFA_ICUadmission + 
                    medical + 
                    shock + 
                    respi2 + 
                    abdominal2 + 
                    uri2 + 
                    others2 +
                    unknown2'  
res[['full']] = test_logbino(imp_base, imp_tab_base, glm_formula, n_covariate=10)



# Confound assessment
# site of infection, medical, shock

# exclude medical
glm_formula = 'dependent_hp_discharge ~ comor+
                    age_ICUadmission + 
                    sex2+
                    SOFA_ICUadmission + 
                    shock + 
                    respi2 + 
                    abdominal2 + 
                    uri2 + 
                    others2 +
                    unknown2'
res[['medical']] =  test_logbino(imp_base, imp_tab_base, glm_formula, n_covariate=9)


# exclude shock
glm_formula = 'dependent_hp_discharge ~ comor +
                    age_ICUadmission + 
                    sex2+
                    SOFA_ICUadmission + 
                    medical+
                    respi2 + 
                    abdominal2 + 
                    uri2 + 
                    others2 +
                    unknown2'
res[['shock']] = test_logbino(imp_base, imp_tab_base, glm_formula, n_covariate=9)


# exclude site of infection
glm_formula = 'dependent_hp_discharge ~ comor+
                    age_ICUadmission + 
                    sex2+
                    SOFA_ICUadmission + 
                    medical+
                    shock'            
res[['soi']] = test_logbino(imp_base, imp_tab_base, glm_formula, n_covariate=5)

res_conf_assess = as_tibble(do.call(rbind, res))
baseRR = unlist(res_conf_assess[1,2])
res_conf_assess %<>% mutate(tenpercentchange = (estimate-baseRR)/baseRR *100)
# keep all covariates



# Each comorbidity using the final model
comor_category  = c('comor','cardiopul','cerebro', 'metabolic', 'cancer')
res = list()
for (i in comor_category){
    glm_formula = paste0('dependent_hp_discharge ~ ',
                         i, 
                         '+ age_ICUadmission +
                          sex2 + 
                          SOFA_ICUadmission + 
                          medical + 
                          shock + 
                          respi2 + 
                          abdominal2 + 
                          uri2 + 
                          others2 +
                          unknown2')  
    res[[i]] = test_logbino(imp_base, imp_tab_base, glm_formula, n_covariate=10)
}    
saveRDS(res, './results/res_base.rds')    
    
    


#---------------------------------------
# Adjusting coraviates and physical impairment before ICU admission without adjusting in-hospital mortality
#--------------------------------------

# study population
pop =  data %>% 
                # filter(is.na(previous_dependence)==T| previous_dependence==0) %>%
                filter(mortality_3day2==0) %>%
                filter(hp_death_after_3days == 0)


# imputation dataset
# exclude bias factors from available variables
impu_base_adj_nu   = numerical
impu_base_adj_cate = c(categorical, 'previous_dependence')
explanatory = c(impu_base_adj_nu, impu_base_adj_cate)
# bias terms were removed because even though this is imputation purpose, study population usually did not include these terms.
dependent = c("dependent_hp_discharge")


# log binomial
# prepare imputation set
cols = c(explanatory, dependent)
temp = pop %>% dplyr::select(all_of(cols))
imp_adj = mice(temp, maxit = 5, m = 10, seed = 1)
imp_tab_adj = complete(imp_adj, 'long') # used for inital values for weights



# Each comorbidity using the final model
comor_category  = c('comor','cardiopul','cerebro', 'metabolic', 'cancer')
res = list()
for (i in comor_category){
    glm_formula = paste0('dependent_hp_discharge ~ ',
                         i, 
                         '+ age_ICUadmission +
                          sex2 + 
                          SOFA_ICUadmission + 
                          medical + 
                          shock + 
                          respi2 + 
                          abdominal2 + 
                          uri2 + 
                          others2 +
                          unknown2 + 
                          previous_dependence')

    res[[i]] = test_logbino(imp_adj, imp_tab_adj, glm_formula, n_covariate=11)
}    
saveRDS(res, './results/res_base_adj.rds')    






#---------------------------
# attrition adjustment
#---------------------------
# study population
pop =  data #%>% 
                # filter(is.na(previous_dependence)==T| previous_dependence==0) %>%
                # filter(mortality_3day2==0) %>%
                # filter(hp_death_after_3days == 0)

# imputation dataset
# exclude bias factors from available variables
impu_att_nu   = numerical
impu_att_cate = c(categorical, 'previous_dependence', 'mortality_3day2', 'hp_death_after_3days')
explanatory = c(impu_att_nu, impu_att_cate)
# bias terms were removed because even though this is imputation purpose, study population usually did not include these terms.
dependent = c("dependent_hp_discharge")


# prepare imputation set
cols = c(explanatory, dependent)
data_temp = pop %>% dplyr::select(all_of(cols))

# for python
write.csv(data_temp, './data/forpython.csv')


# Probability of in-hospital death that were estimated by python
prob_3day = read.table('PS_deathIn72hours.txt', sep = ',', header = T, col.names= c('x', 'prob'))
prob_hp   = read.table('PS_deathInHpAfter72hours.txt', sep = ',', header = T, col.names= c('x', 'prob'))
probs = tibble(prob_3day = prob_3day$prob,
              prob_hp = prob_hp$prob)

# PS 
probs %<>% mutate(prob_not_3day = 1- prob_3day) 
probs %<>% mutate(prob_not_hp = 1- prob_hp)

# marginal probability of attrition factors
margeP_died_72hrs = sum(temp$mortality_3day2)/nrow(temp)
margeP_not_died_72hrs = 1 - margeP_died_72hrs

margeP_died_hp_after72hrs = sum(temp$hp_death_after_3days)/nrow(temp)
margeP_not_died_hp_after72hrs = 1 - margeP_died_hp_after72hrs

# stabilized weights
probs %<>% mutate(stawts = (margeP_not_died_72hrs*margeP_not_died_hp_after72hrs)/(prob_not_3day*prob_not_hp))


# add stab weights
combine = data_temp %>% mutate(stabwts = probs$stawts)
n_samples  = nrow(combine)


# imputation matrix, exclude the weights from imputer 
predM = combine %>% missing_predictorMatrix(drop_from_imputer = c('stabwts'))
imp_adj = mice(combine, maxit = 5, m = 10, seed = 1,predictorMatrix = predM )
imp_tab_adj = complete(imp_adj, 'long') # used for inital values for weights

test_logbino_final = function(imp_mice, mice_table, glm_formula, n_covariate){
        fit = with(imp_mice, glm(formula(glm_formula), 
                        family = binomial(link='log'),
                        weights = mice_table$stabwts[1:n_samples],
                        start=c(log(mean(mice_table$dependent_hp_discharge)), rep(0, n_covariate+1))))  
        sum_ = summary(pool(fit), conf.int=T, exponentiate=T) 
        res = sum_[2,]
        return(res) 
}


# Each comorbidity using the final model
comor_category  = c('comor','cardiopul','cerebro', 'metabolic', 'cancer')
res = list()
for (i in comor_category){
    glm_formula = paste0('dependent_hp_discharge ~ ',
                         i, 
                         '+ age_ICUadmission +
                          sex2 + 
                          SOFA_ICUadmission + 
                          medical + 
                          shock + 
                          respi2 + 
                          abdominal2 + 
                          uri2 + 
                          others2 +
                          unknown2 + 
                          previous_dependence')
    res[[i]] = test_logbino_final(imp_adj, imp_tab_adj, glm_formula, n_covariate=11)
}    

saveRDS(res, './results/res_att_adj.rds')    


#--------------------------------
# visualization
#--------------------------------
uni = readRDS('./results/res_univariate.rds')   
base = readRDS('./results/res_base_adj.rds')
adj = readRDS('./results/res_base.rds')
att = readRDS('./results/res_att_adj.rds')

uni1 = as_tibble(do.call(rbind,uni))
base1 = as_tibble(do.call(rbind,base))
adj1 = as_tibble(do.call(rbind,adj))
att1 = as_tibble(do.call(rbind,att))

uni1 %<>% mutate(adj = 'uni')
base1 %<>% mutate(adj = 'confs')
adj1 %<>% mutate(adj = 'base')
att1 %<>% mutate(adj = 'att')

combine = rbind(uni1, base1, adj1, att1)
names(combine) = c("term", "estimate", "std.error",
                 "statistic", "df", "p.value", 
                 "lower"  , "upper",  "adj")    

combine$adj = factor(combine$adj, levels =c('att', 'uni','confs','base'))
levels(combine$adj) = c("Primary model", "Sub model 1", "Sub model 2",'Sub model 3')

combine$term = factor(combine$term, 
        levels =c('comor','cardiopul','cerebro','metabolic','cancer'))
levels(combine$term) = c("Comorbodity","Cardiopulmonary",
                              "Cerebrovascular","Metabolic",
                              "Cancer")


final = combine %>% ggplot(aes(x = term, y = estimate, color = adj)) + 
            geom_point(shape=18,size=3, 
               position=position_dodge(preserve = "total",width = 0.4)) +
            geom_errorbar(aes(ymin =lower,ymax=upper), 
                width=0.2,
                position=position_dodge(preserve = "total",width = 0.4))+
            scale_color_discrete(name = '')+
            theme_bw() +
            xlab("")+
            ylab("Risk ratio") +
            theme(axis.text.x = element_text(size=12,angle=0,hjust = 0.5,vjust=0.6),
                  axis.text.y = element_text(size=12),
                  axis.title=element_text(size=14))+
            geom_hline(yintercept = 1.0, linetype = 'dashed', color="red",linewidth=0.5)

ggsave('rr_plot.pdf', final, width = 10, height = 6, dpi=400, units = 'in')
