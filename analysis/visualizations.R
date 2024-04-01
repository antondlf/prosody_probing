library(ggplot2)
library(dplyr)
library(ggthemes)
library(stringr)
library(ggpubr)
library(RColorBrewer)
library(scales)
setwd('/Users/anton/Projects/prosody_probing/analysis')

plot_mtwvs <- function(mtwv_df, colors, wav2vec_checkpoint_name = "wav2vec 2.0 Large (LibriSpeech 960h)"){

  plot_df <- mtwv_df %>%
    rename(
      Model  = model,
      Layer = layer,
      Score     = score,
      Language = Language
    ) %>%
    mutate(
      #Model = case_when(
      #  Model %in% c("random", "fbank")    ~ str_to_upper(Model),
      #),
#      Dataset = ifelse(
#        Dataset %in% c("FBANK", "RANDOM"),
#        Dataset,
#        Dataset
#      ), #%>%
      # Legend ordering
      #Model = factor(
      #  Model,
      #  levels = c("wav2vec2-base", "mandarin-wav2vec2", "mandarin-wav2vec2-aishell1", "wav2vec2-base-100h", "wav2vec2-large", "wav2vec2-xls-r-300m", "wav2vec2-large-xlsr-53", "hubert-base-ls960", "wavlm-base", 'random', 'fbank') ),
      State = factor(
        State,
        levels = c("Pre-trained", "Fine-tuned", "Baseline")
      ),
      )
  # Renaming plot_df the same name is ripe for variable name conflicts
  # But alas I am lazy and it works

  baseline_df <- plot_df[(plot_df$Model %in% c('random')),]
  print(baseline_df)
  plot_df <- plot_df[!(plot_df$Model %in% c('random', 'fbank', 'mfcc')),]
  
  plot_top_ranked <- plot_df %>% 
    group_by(task_name, Model) %>%
    slice_max(order_by = Score, n = 1)

  plot_df %>% 
    ggplot(aes(x = Layer, y = Score, group = Model)) +
    geom_hline(data=baseline_df, aes(yintercept=Score), color='red', linetype='dashed', show.legend=FALSE)  +
    geom_line(aes(color = colColor, linetype=State), linewidth=2, show.legend=TRUE) +
    # c('English', 'Mandarin')c('wav2vec 2.0', 'HuBERT', 'WavLM')
    scale_color_identity(name = 'Model', label=c('English', 'Mandarin'), guide=guide_legend()) +
    geom_point(aes(shape = Model, color=colColor), size = 6, show.legend=FALSE) +
    #geom_ribbon(aes(ymin=ci_low, ymax=ci_high), linetype=2, alpha=0.1) +
    geom_point(
      aes(shape = Model),
      size = 7,
      stroke = 1,
      color = "#e41a1c",
      data = plot_top_ranked,
      show.legend = FALSE
      # position = "jitter"
    ) +
    facet_grid(~task_name,space = "free_x", scales = "free_x") +
    scale_y_continuous(limits = c(0, 1)) +
    theme(legend.position="bottom") +
    theme_base(base_size = 14) +
    theme(
      rect = element_blank(),
      panel.grid.major = element_line(colour = "grey"),
      strip.text.x = element_text(size = 30),
      axis.text=element_text(size=30),
      axis.title=element_text(size=30),
      legend.position='bottom',
      legend.margin=margin(0,0,0,0),
      legend.box.margin=margin(-5,-5,-5,-5),
      legend.text=element_text(size=30),
      legend.title=element_text(size=30)
    ) +
    labs(y='F1 Score')+
    #ylim(0.25, 1) +
    # scale_shape_manual(values = c(
    #   18, # eng-mav
    #   19, # nld-gng
    #   2,  # gbb-pd
    #   6,  # wrm-pd
    #   3,  # wrl-mb
    #   4,  # gup-wat
    #   8,  # pjt-sw01
    #   1,  # wbp-jk
    #   0,  # mwf-jm
    #   5   # gbb-lg
    # )) +
    #scale_fill_manual(values=color.set) +
    #colScale +
    #scale_color_colorblind(drop=TRUE) +
    scale_x_continuous (breaks=pretty_breaks()) +
    guides(
      colour = guide_legend(show = TRUE, override.aes = list(size=10, width=60)),
      linetype = "none",
      shape = "none"#guide_legend(override.aes = list(size = 4), nrow = 2)
    ) 
}
data <- read.csv('../results/interspeech_final_results.csv')
#data <- read.csv('../results/multi_metric_results.csv')
#data2 <- read.csv('../results/multi_metric_results_final.csv')
data.cols <- names(data)
#data$source <- 'first pass'
#data2$source <- 'second pass'
#data <- merge(x=data, y=data2, all=TRUE, by=data.cols)
#data$rowSource <- apply(data[c("source.x", "source.y")], 1, 
#                          function(x) paste(na.omit(x), collapse #= ""))
#data <- data[(data$rowSource == 'first pass')|(data$model == 'random'),]
factor.cols <- c("model", "model_names", "model_state", "model_size", "language", "probe", "task", "corpus", "task_name", "metric")

num.cols <- c("score")
int.cols <- c("layer")
data[factor.cols] <- lapply(data[factor.cols], as.factor)
data[num.cols] <- lapply(data[num.cols], as.numeric)
data[int.cols] <- lapply(data[int.cols], as.integer)

layer_sizes <- c("12", "24", "0")
names(layer_sizes) <- c("Base", "Large", "Baseline")
layer_sizes['Base']

get_model_size <- function(string){
  if (string == 'Base'){
    return(12)
  }
  else if (string == 'Large'){
    return(24)
  }
  else{
    return(1)
  }
}

#data <- data %>% mutate(layer = case_when(model_size %in% c('Large') ~ layer / 2, model_size %in% c('Base') ~ layer))

data["max_layer"] <- sapply(data$model_size, get_model_size)

data['model_percent'] <- (data$layer / data$max_layer)*12
#data['Model'] <- data$model_names
data['State'] <- data$model_state
data['Language'] <- data$corpus
data['Comparison'] <- data$model_names
data <- data %>%mutate(Language = case_when(corpus %in% c('switchboard') ~ 'English', corpus %in% c('mandarin-timit') ~ 'Mandarin'))

monolingual_models <- c('English Large', 'English Base', 'Mandarin Base', 'English Finetuned', 'Mandarin Finetuned', 'English Large Finetuned', 'English Large Finetunted', 'random')
multilingual_models <- c('Multilingual Large 128', 'Multilingual Large 53', 'random')
monolingual_base <- c('English Base', 'Mandarin Base', 'random')
monolingual_finetuned <- c('English Finetuned', 'Mandarin Finetuned', 'English Large Finetuned', 'English Large Finetunted', 'random')
other_models <- c('HuBert', 'WavLM', 'random')

final_models <-  c('English Base', 'Mandarin Base', 'English Finetuned', 'Mandarin Finetuned', 'Random Baseline', 'HuBert', 'WavLM')

data <- data[(data$model_names %in% final_models),]
data$model <- factor(data$model)

data <- data %>% mutate(Language = recode(language, Mandarin = 'Mandarin', English = 'English'))

data <- data %>% mutate(Comparison = case_when(model_names %in% c('HuBert', 'WavLM') ~ 'Other Models', model_names %in% monolingual_models ~ 'Monolingual wav2vec 2.0', model_names %in% multilingual_models ~ 'Multilingual wav2vec 2.0')) # model_names %in% c('English Finetuned', 'Mandarin Finetuned', 'English Large Finetuned', 'English Large Finetunted') ~ 'Fine-tuned'),)

data$task_name <- factor(data$task, levels=c("syllables_accents", "stress", "tone", "f0", "f0_300", "stress_polysyllabic", "energy", "intensity"), labels=c("English Accent", "English Stress", "Mandarin Tone", "F0", "F0", "English Stress", "RMS Energy", "Intensity (dB SPL)"),)




plot_df <- data[((data$metric == 'f1')|(data$metric == 'R2')) &!(data$model_names %in% monolingual_finetuned)&(data$corpus != 'yemba')&(data$corpus != 'bu_radio')&(data$probe == 'linear') & !(data$model %in% c('wav2vec2-large', 'wav2vec2-large-960h', 'hubert-base-ls960', 'wavlm-base', 'wav2vec2-xls-r-300m', 'wav2vec2-large-xlsr-53')) & ((data$task %in% c('stress', 'syllables_accents', 'tone')) | ((data$task == 'f0') & (data$corpus == 'switchboard'))),]

colors <- data.frame(Language =c("Mandarin", "English"),
                       colColor=c("#E69F00", "#000000"))

plot_df <- plot_df %>% left_join(colors, by = "Language")

plot_df2 <- data[((data$metric == 'f1')|(data$metric == 'R2')) &((data$model_names %in% monolingual_models)|(data$model == 'random'))&(data$corpus != 'yemba')&(data$corpus != 'bu_radio')&(data$probe == 'linear')&((data$model != 'wav2vec2-large') & (data$model != 'wav2vec2-large-960h'))&((data$task == 'tone')|(data$task == 'syllables_accents')|(data$task == 'stress')),]

colors2 <- data.frame(Language =c("Mandarin", "English"),
                       colColor=c("#E69F00", "#000000"))

plot_df2 <- plot_df2 %>% left_join(colors2, by = "Language")

plot_df3 <- data[((data$metric == 'f1')|(data$metric == 'R2')) &!(data$model_names %in% monolingual_finetuned)&(data$corpus != 'yemba')&(data$corpus != 'bu_radio')&(data$probe == 'linear')& (data$model %in% c('wav2vec2-base', 'hubert-base-ls960', 'wavlm-base', 'random')) & ((data$task %in% c('stress', 'syllables_accents', 'tone'))),]# | ((data$task == 'f0') & (data$corpus == 'switchboard'))),]

colors3 <- data.frame(model_names =c("wav2vec 2.0", "HuBERT", 'WavLM'),
                       colColor=c("#000000", "#D55E00", "#0072B2"))

plot_df3$model_names <- with(plot_df3, factor(model, 
    levels = c('wav2vec2-base', 'hubert-base-ls960', 'wavlm-base'), labels = c('wav2vec 2.0', 'HuBERT','WavLM')))

plot_df3 <- plot_df3 %>% left_join(colors3, by = "model_names")


plot_df %>% plot_mtwvs(colors, wav2vec_checkpoint_name = "Layerwise Linear Probe Performance")

ggsave('paper_plots/all_feats_wav2vec2_final.png', height=10, width=15)
#ggsave('paper_plots/all_feats_hubert_wavlm_final.png', height=10, width=15)
#ggsave('paper_plots/all_feats_finetuning_final.png', height=10, width=15)

