namespace = recog/mnist_tiny
dataset = mnist
input_shape = 1 28 28
model_name = mini
batch_size = 4
val_batch_size = 4
samples_per_epoch = 20
val_samples_per_epoch = 20
epochs = 2
nsamples = 10
max_plots = 10
log_interval = 1

results = /opt/data/results
path = $(results)/$(namespace)

plot_data :
	python -m pt.recog.tasks.plot_data \
		--namespace recog \
		--dataset $(dataset) \
		--split train \
		--nimages 12

$(path)/save_gt/%.npy :
	python -m pt.recog.tasks.save_gt \
		--namespace $(namespace) \
		--dataset $(dataset) \
		--split $*

save_train_gt : $(path)/save_gt/train.npy
save_val_gt : $(path)/save_gt/val.npy

$(path)/train_model/done :
	python -m pt.recog.tasks.train_model \
		--namespace $(namespace) \
		--dataset $(dataset) \
		--input-shape $(input_shape) \
		--batch-size $(batch_size) \
		--val-batch-size $(val_batch_size) \
		--samples-per-epoch $(samples_per_epoch) \
		--val-samples-per-epoch $(val_samples_per_epoch) \
		--epochs $(epochs) \
		--log-interval $(log_interval)
	touch $(path)/train_model/done

train_model : $(path)/train_model/done

$(path)/plot_log/plot.png : $(path)/train_model/done
	python -m pt.recog.tasks.plot_log \
		--namespace $(namespace)

plot_log : $(path)/plot_log/plot.png

$(path)/infer_probs/%.npy : $(path)/train_model/done
	python -m pt.recog.tasks.infer_probs \
		--namespace $(namespace) \
		--dataset $(dataset) \
		--model-name $(model_name) \
		--input-shape $(input_shape) \
		--split $* \
		--nsamples $(nsamples)

infer_train_probs : $(path)/infer_probs/train.npy
infer_val_probs : $(path)/infer_probs/val.npy
infer_test_probs : $(path)/infer_probs/test.npy

$(path)/infer_preds/%.npy : $(path)/infer_probs/%.npy
	python -m pt.recog.tasks.infer_preds \
		--namespace $(namespace) \
		--split $*

infer_train_preds : $(path)/infer_preds/train.npy
infer_val_preds : $(path)/infer_preds/val.npy
infer_test_preds : $(path)/infer_preds/test.npy

$(path)/compute_scores/scores.json : $(path)/infer_preds/val.npy $(path)/save_gt/val.npy
	python -m pt.recog.tasks.compute_scores \
		--namespace $(namespace)

compute_scores : $(path)/compute_scores/scores.json

$(path)/plot_preds/done : $(path)/infer_preds/val.npy $(path)/save_gt/val.npy
	python -m pt.recog.tasks.plot_preds \
		--namespace $(namespace) \
		--dataset $(dataset) \
		--max-plots $(max_plots)
	touch $(path)/plot_preds/done

plot_preds : $(path)/plot_preds/done

all : compute_scores plot_log plot_preds

clean :
	rm -R $(path)

.PHONY:
	plot_data
	save_train_gt
	save_val_gt
	train_model
	plot_log
	infer_train_probs
	infer_val_probs
	infer_test_probs
	infer_train_preds
	infer_val_preds
	infer_test_preds
	compute_scores
	plot_preds
	all
	clean
