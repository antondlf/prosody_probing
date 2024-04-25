library(ggplot2)
library(dplyr)
library(ggthemes)
library(stringr)
library(ggpubr)
setwd('/Users/anton/Projects/prosody_probing/analysis')
color.set <- c('wav2vec2-base' = "#56B4E9", 'mandarin-wav2vec2' = "#D55E00", 'HuBert' = "", "WavLM" = "")


single_plot <- function(plot_df, baseline_df, plot_top_ranked){
    ggplot(aes(x = Layer, y = Score, group = Model)) +
    geom_hline(data=baseline_df, aes(yintercept=Score), color='red', linetype='dashed', show.legend=FALSE)  +
    geom_line(aes(color = Model), size=1.25, show.legend=TRUE) +
    geom_point(aes(shape = Model), size = 2, show.legend=FALSE) +
    geom_point(
      aes(shape = Model),
      size = 1.5,
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
      legend.position='bottom',
      legend.margin=margin(0,0,0,0),
      legend.box.margin=margin(-5,-5,-5,-5)
    ) +
    #ylim(0.25, 1) +
    scale_shape_manual(values = c(
      18, # eng-mav
      19, # nld-gng
      2,  # gbb-pd
      6,  # wrm-pd
      3,  # wrl-mb
      4,  # gup-wat
      8,  # pjt-sw01
      1,  # wbp-jk
      0,  # mwf-jm
      5   # gbb-lg
    )) +
    #scale_fill_manual(values=color.set) +
    scale_color_colorblind() +
    scale_x_discrete(guide = guide_axis(n.dodge=3)) +
    guides(
      colour = guide_legend(show = TRUE),
      shape = guide_legend(override.aes = list(size = 4), nrow = 2)
    ) 
}
plot_mtwvs <- function(mtwv_df, color.set, wav2vec_checkpoint_name = "wav2vec 2.0 Large (LibriSpeech 960h)"){

  plot_df <- mtwv_df %>%
    rename(
      Model  = model,
      Layer = model_percent,
      Score     = score
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
      Model = factor(
        Model,
        levels = c("wav2vec2-base", "mandarin-wav2vec2", "mandarin-wav2vec2-aishell1", "wav2vec2-base-100h", "wav2vec2-large", "wav2vec2-xls-r-300m", "wav2vec2-large-xlsr-53", "hubert-base-ls960", "wavlm-base", 'random', 'fbank') ),
      State = factor(
        State,
        levels = c("Pre-trained", "Fine-tuned", "Baseline")
      ),
      )
  # Renaming plot_df the same name is ripe for variable name conflicts
  # But alas I am lazy and it works

  baseline_df1 <- plot_df[(plot_df$Model %in% c('random'))&(data$task == 'tone'),]
  baseline_df2 <- plot_df[(plot_df$Model %in% c('random'))&(data$task == 'syllables_accents'),]
  baseline_df3 <- plot_df[(plot_df$Model %in% c('random'))&(data$task == 'stress'),]
  plot_df1 <- plot_df[!(plot_df$Model %in% c('random', 'fbank', 'mfcc')) &(data$task == 'tone'),]
  plot_df2 <- plot_df[!(plot_df$Model %in% c('random', 'fbank', 'mfcc')) &(data$task == 'syllables_accents'),]
  plot_df3 <- plot_df[!(plot_df$Model %in% c('random', 'fbank', 'mfcc')) &(data$task == 'stress'),]
  
  plot_top_ranked <- plot_df %>% 
    group_by(task_name, Model) %>%
    slice_max(order_by = Score, n = 1)
  
  p1 <- single_plot(plot_df1, baseline_df1, plot_top_ranked)
  p2 <- single_plot(plot_df2, baseline_df2, plot_top_ranked)
  p3 <- single_plot(plot_df3, baseline_df3, plot_top_ranked)
  
  grid.arrange(p1, p2, p3, layout_matrix = matrix(c(1, 3, 2, 3), nrow = 2)) 
}

data <- read.csv('../results/multi_metric_results.csv')
data2 <- read.csv('../results/multi_metric_results_final.csv')
data.cols <- names(data)
data$source <- 'first pass'
data2$source <- 'second pass'
data <- merge(x=data, y=data2, all=TRUE, by=data.cols)
data$rowSource <- apply(data[c("source.x", "source.y")], 1, 
                          function(x) paste(na.omit(x), collapse = ""))
data <- data[(data$rowSource == 'first pass')|(data$model == 'random'),]
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

data <- data %>% mutate(Comparison = case_when(model_names %in% c('HuBert', 'WavLM') ~ 'Other Models', model_names %in% monolingual_models ~ 'Monolingual wav2vec 2.0', model_names %in% multilingual_models ~ 'Multilingual wav2vec 2.0')) # model_names %in% c('English Finetuned', 'Mandarin Finetuned', 'English Large Finetuned', 'English Large Finetunted') ~ 'Fine-tuned'),)

data$task_name <- factor(data$task, levels=c("syllables_accents", "stress", "tone", "f0", "f0_300", "stress_polysyllabic", "energy", "intensity"), labels=c("Accent", "Stress", "Tone", "F0", "F0", "Stress", "RMS Energy", "Intensity (dB SPL)"),)

plot_df <- data[((data$metric == 'f1')|(data$metric == 'R2')) &!(data$model_names %in% monolingual_finetuned)&(data$corpus != 'yemba')&(data$corpus != 'bu_radio')&(data$probe == 'linear') & !(data$model %in% c('wav2vec2-large', 'wav2vec2-large-960h', 'hubert-base-ls960', 'wavlm-base', 'wav2vec2-xls-r-300m', 'wav2vec2-large-xlsr-53')) & ((data$task %in% c('stress', 'syllables_accents', 'tone')) | ((data$task == 'f0') & (data$corpus == 'switchboard'))),]

plot_df2 <- data[((data$metric == 'f1')|(data$metric == 'R2')) &(data$model_names %in% monolingual_models)&(data$model == 'random')&(data$corpus != 'yemba')&(data$corpus != 'bu_radio')&(data$probe == 'linear')&(data$model_type != 'Baseline')&((data$model != 'wav2vec2-large') & (data$model != 'wav2vec2-large-960h'))&((data$task == 'tone')|(data$task == 'syllables_accents')|(data$task == 'stress')|((data$task %in% c('f0', 'intensity')) & (data$corpus == 'switchboard'))),]

plot_df3 <- data[((data$metric == 'f1')|(data$metric == 'R2')) &!(data$model_names %in% monolingual_finetuned)&(data$corpus != 'yemba')&(data$corpus != 'bu_radio')&(data$probe == 'linear')&(data$model_type != 'Baseline') & (data$model %in% c('wav2vec2-base', 'hubert-base-ls960', 'wavlm-base', 'random')) & ((data$task %in% c('stress', 'syllables_accents', 'tone'))),]# | ((data$task == 'f0') & (data$corpus == 'switchboard'))),]

plot_df %>% plot_mtwvs(wav2vec_checkpoint_name = "Layerwise Linear Probe Performance")

