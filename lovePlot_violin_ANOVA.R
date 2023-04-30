

# This code is used to summarize the results from PS estimation using python. This includes love plots of standardized mean difference, violin plots of the Area Under the Receiver Operating Curve (AUROC) for predicting attrition factors. An ANOVA test was performed to compare AUROC across machine learning models. 


#------------------------------------------
# required packages
#------------------------------------------
library(tidyverse)
library(magrittr)
library(ggpubr)
library(egg)


#----------------------------------------
# names used for visualization
#---------------------------------------
model_ = c('log','xgb','nn')
model = list('Logistic regression','XGBoost','Neural Network')
covars =c("age_ICUadmission", "SOFA_ICUadmission",
            "sex2", "previous_dependence", "medical", "shock",
            "comor", "cardiopul", "cerebro", "metabolic", "cancer",
            "respi2", "abdominal2","uri2","others2","unknown2" ,
            "blood_culture")
covars_rename =c("Age", "SOFA score at ICU admission",
            "Female", "Physical dependence prior to ICU admission",
             "Medical", "Shock",
            "Comorbidity", "Cardiopulmonary comorbidity", "Cerebrovascular comorbidity",
            "Metabolic comorbidity", "Cancer comorbidity",
            "Respiratory infection", "Abdominal infection","Urinary tract infection",
            "Other infection","Unknown source infection" ,
            "Blood culture positive")



# -------------------------------------
# load data
# -------------------------------------

roc_3day = list()
roc_3day[['log']] = read.table('roc_log.txt', sep = ',', header = T, col.names = c('imp_cv','roc'))
roc_3day[['xgb']] = read.table('roc_xgb.txt', sep = ',', header = T, col.names = c('imp_cv','roc'))
roc_3day[['nn']] = read.table('roc_nn.txt', sep = ',', header = T, col.names = c('imp_cv','roc'))


roc_hp = list()
roc_hp[['log']] = read.table('roc_log_hp.txt', sep = ',', header = T, col.names = c('imp_cv','roc'))
roc_hp[['xgb']] = read.table('roc_xgb_hp.txt', sep = ',', header = T, col.names = c('imp_cv','roc'))
roc_hp[['nn']] = read.table('roc_nn_hp.txt', sep = ',', header = T, col.names = c('imp_cv','roc'))


smd_3day = list()
smd_3day[['log']] = read.table('smd_log.txt',sep = ',', header = T)
smd_3day[['xgb']] = read.table('smd_xgb.txt',sep = ',', header = T) 
smd_3day[['nn']]  = read.table('smd_nn.txt',sep = ',', header = T) 

smd_hp = list()
smd_hp[['log']] = read.table('smd_log_hp.txt',sep = ',', header = T)
smd_hp[['xgb']] = read.table('smd_xgb_hp.txt',sep = ',', header = T) 
smd_hp[['nn']]  = read.table('smd_nn_hp.txt',sep = ',', header = T) 




# ----------------------------------
# love plots of SMD
# ----------------------------------

plots_3day = mapply(FUN = function(data,models){
    temp = data
    # make long table
    temp$var = factor(temp$var, levels = covars)
    levels(temp$var) = covars_rename
    temp$var = fct_rev(temp$var)
    temp1 = pivot_longer(temp, cols = c('smd', 'wt_smd'), names_to = 'group', values_to = 'value')
    text = paste0(models)
    temp1$group = factor(temp1$group, levels = c('smd', 'wt_smd'))
    levels(temp1$group) = c('Non-weighted SMD', 'Weighted SMD')
    plot = temp1 %>% ggplot(aes(x=var, y = value, color = group)) + 
                        geom_point()+
                        geom_line(aes(group = group))+
                        coord_flip()+
                        theme_bw()+
                        xlab('')+
                        ylab('Standardized Mean Difference')+
                        scale_color_discrete(name = '')+
                        ggtitle(text)+
                        geom_hline(yintercept = 0.1, color = 'red', 
                                linetype = 'dashed', alpha = 1, linewidth = 0.3)+
                        theme(
                            axis.text=element_text(size=12),
                            axis.title=element_text(size=12)
                        )
    return(plot)
    }, smd_3day, model,
    SIMPLIFY = F
)




plots_hp = mapply(FUN = function(data,models){
    temp = data
    # make long table
    temp$var = factor(temp$var, levels = covars)
    levels(temp$var) = covars_rename
    temp$var = fct_rev(temp$var)
    temp1 = pivot_longer(temp, cols = c('smd', 'wt_smd'), names_to = 'group', values_to = 'value')
    text = paste0(models)
    temp1$group = factor(temp1$group, levels = c('smd', 'wt_smd'))
    levels(temp1$group) = c('Non-weighted SMD', 'Weighted SMD')
    plot = temp1 %>% ggplot(aes(x=var, y = value, color = group)) + 
                        geom_point()+
                        geom_line(aes(group = group))+
                        coord_flip()+
                        theme_bw()+
                        xlab('')+
                        ylab('Standardized Mean Difference')+
                        scale_color_discrete(name = '')+
                        ggtitle(text)+ 
                        geom_hline(yintercept = 0.1, color = 'red', 
                                linetype = 'dashed', alpha = 1, linewidth = 0.3)+
                        theme(
                            axis.text=element_text(size=12),
                            axis.title=element_text(size=12)
                        )+
                        ylim(0,0.5)

    return(plot)
    }, smd_hp, model,
    SIMPLIFY = F
)


#----------------------------------------
# Violin plots of AUROC and ANOVA test
#----------------------------------------

vioplots = list()
# death within 72 hours
data = tibble(log = roc_3day[['log']]$roc, 
              xgb = roc_3day[['xgb']]$roc,
              nn= roc_3day[['nn']]$roc)

# long format
temp = pivot_longer(data, cols = c('log','xgb','nn'), 
                        names_to = 'model', values_to = 'roc')  
# factorize and rename
temp$model = factor(temp$model, levels = c('log','xgb','nn'))
levels(temp$model) = unlist(model)
# plot
vioplots[['3day']] = temp %>% ggplot(aes(model, roc)) +
            geom_violin(aes(fill = model), alpha = 0.4)+
            geom_jitter(aes(color = model), width =0.3)+ 
            theme_bw()+
            xlab('')+
            ylab('Area Under Receiver Operating Curve')+
            scale_color_discrete(name = 'Model')+
            scale_fill_discrete(name = 'Model')


temp %>% group_by(model) %>% summarise(mean = mean(roc))
#   model                mean
#   <fct>               <dbl>
# 1 Logistic regression 0.802
# 2 XGBoost             0.969
# 3 Neural Network      0.847

# anova
fit = aov(roc ~model, data = temp)
TukeyHSD(fit)

# Tukey multiple comparisons of means
# 95% family-wise confidence level
# 
# $model
#                                           diff        lwr         upr p adj
# XGBoost-Logistic regression         0.16686357  0.1492592  0.18446795 0e+00
# Neural Network-Logistic regression  0.04568058  0.0280762  0.06328495 2e-06
# Neural Network-XGBoost             -0.12118300 -0.1387874 -0.10357862 0e+00




# death in a hospital after 72 hours
data = tibble(log = roc_hp[['log']]$roc, 
              xgb = roc_hp[['xgb']]$roc,
              nn=   roc_hp[['nn']]$roc)

# long
temp = pivot_longer(data, cols = c('log','xgb','nn'), 
                        names_to = 'model', values_to = 'roc')  
# factorize and rename
temp$model = factor(temp$model, levels = c('log','xgb','nn'))
levels(temp$model) = unlist(model)
# plot
vioplots[['hp']] = temp %>% ggplot(aes(model, roc)) +
            geom_violin(aes(fill = model), alpha = 0.4)+
            geom_jitter(aes(color = model), width =0.3)+ 
            theme_bw()+
            xlab('')+
            ylab('Area Under Receiver Operating Curve')+
            scale_color_discrete(name = 'Model')+
            scale_fill_discrete(name = 'Model')

temp %>% group_by(model) %>% summarise(mean = mean(roc))
#   model                mean
#   <fct>               <dbl>
# 1 Logistic regression 0.724
# 2 XGBoost             0.879
# 3 Neural Network      0.731


# anova
fit = aov(roc ~model, data = temp)
TukeyHSD(fit)

# Tukey multiple comparisons of means
# 95% family-wise confidence level
# $model
#                                            diff          lwr         upr     p adj
# XGBoost-Logistic regression         0.154591402  0.148277817  0.16090499 0.0000000
# Neural Network-Logistic regression  0.006750491  0.000436906  0.01306408 0.0343259
# Neural Network-XGBoost             -0.147840911 -0.154154496 -0.14152733 0.0000000



# output plots
violins = egg::ggarrange(plots = vioplots, ncol = 2, labels = c("A",'B'),font = 5)
violins = ggpubr::ggarrange(vioplots[['3day']],vioplots[['hp']], ncol = 2, labels = c("A",'B'))

love_3day = egg::ggarrange(plots = plots_3day, ncol = 1)
love_hp = egg::ggarrange(plots = plots_hp, ncol = 1)
loves = ggpubr::ggarrange(love_3day, love_hp, nrow = 1, labels = c('A','B'))

# save
ggsave('violin.pdf', violins, width= 15, height=10, dpi=400, units ='in')
ggsave('love_3day.pdf', love_3day, width= 10, height=20, dpi=400, units ='in')
ggsave('love_hp.pdf', love_hp, width= 10, height=20, dpi=400, units ='in')
ggsave('loves.pdf', loves, width= 17, height=17, dpi=400, units ='in')

