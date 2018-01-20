/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.murdodepar.test;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Random;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.SavedModelBundle;

public class Test {
	public static void main(String[] args) throws Exception {
		if (args.length != 2) {
			System.err.println("Require two arguments: <graph_def_filename> <directory_for_checkpoints>");
			System.exit(1);
		}
		if(!Files.exists(Paths.get(args[0]))){
			System.out.println("Pas de graph");
			try (GraphUtil gu = new GraphUtil();) {
				//# Batch of input and target output (1x1 matrices)
				//x = tf.placeholder(tf.float32, shape=[None, 1, 1], name='input')
				Output<OperationBuilder> x = gu.setPlaceholder("input", DataType.FLOAT);
				//y = tf.placeholder(tf.float32, shape=[None, 1, 1], name='target')
				Output<OperationBuilder> y = gu.setPlaceholder("target", DataType.FLOAT);
				//# Trivial linear model
				//y_ = tf.identity(tf.layers.dense(x, 1), name='output')
				Output<OperationBuilder> y_ = gu.identity(x, "output");
				//# Optimize loss
				//loss = tf.reduce_mean(tf.square(y_ - y), name='loss')
				Output<OperationBuilder> sq = gu.square(gu.setSub("sub", y_, y), "loss");
				Output<OperationBuilder> loss = gu.reduceMean("loss",sq);
			//optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
			//train_op = optimizer.minimize(loss, name='train')

			//init = tf.global_variables_initializer()

				//# tf.train.Saver.__init__ adds operations to the graph to save
				//# and restore variables.
				//saver_def = tf.train.Saver().as_saver_def()
				gu.saveGraphDef(args[0]);
		    
				gu.loadGraphDef(args[0]);
			}
			return;
		}
		final byte[] graphDef = Files.readAllBytes(Paths.get(args[0]));
		final String checkpointDir = args[1];
		final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

		// These names of tensors/operations in the graph (string arguments to feed(),
		// fetch(), and
		// addTarget()) would have been printed out by model.py
		try (
				Graph graph = new Graph();
				Session sess = new Session(graph);
				Tensor<String> checkpointPrefix = Tensors.create(Paths.get(checkpointDir, "checkpoint").toString())
		) {
			graph.importGraphDef(graphDef);

			// Initialize or restore.
			if (checkpointExists) {
				System.out.println("Restoring variables from checkpoint");
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
			} else {
				System.out.println("Initializing variables");
				sess.runner().addTarget("init").run();
			}
			System.out.println("Generating initial predictions");
			printPredictionsOnTestSet(sess);

			System.out.println("Training for a few steps");
			final int BATCH_SIZE = 10;
			float inputs[][][] = new float[BATCH_SIZE][1][1];
			float targets[][][] = new float[BATCH_SIZE][1][1];
			for (int i = 0; i < 200; ++i) {
				fillNextBatchForTraining(inputs, targets);
				try (
					Tensor<Float> inputBatch = Tensors.create(inputs);
					Tensor<Float> targetBatch = Tensors.create(targets)
				) {
					sess.runner().feed("input", inputBatch).feed("target", targetBatch).addTarget("train").run();
				}
			}

			System.out.println("Updated predictions");
			printPredictionsOnTestSet(sess);

			System.out.println("Saving checkpoint");
			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
		}
	}

	public static void printPredictionsOnTestSet(Session sess) {
		final float[][][] inputBatch = new float[][][] { { { 1.0f } }, { { 2.0f } }, { { 3.0f } } };
		try (Tensor<Float> input = Tensors.create(inputBatch);
				Tensor<Float> output = sess.runner().feed("input", input).fetch("output").run().get(0)
						.expect(Float.class)) {
			final long shape[] = output.shape();
			final int batchSize = (int) shape[0];
			final int rows = (int) shape[1];
			final int cols = (int) shape[2];
			float[][][] predictions = output.copyTo(new float[batchSize][rows][cols]);
			for (int i = 0; i < batchSize; ++i) {
				System.out.print("\t x = ");
				System.out.print(Arrays.deepToString(inputBatch[i]));
				System.out.print(", predicted y = ");
				System.out.println(Arrays.deepToString(predictions[i]));
			}
		}
	}

	public static void fillNextBatchForTraining(float[][][] inputs, float[][][] targets) {
		final Random r = new Random();
		for (int i = 0; i < inputs.length; ++i) {
			inputs[i][0][0] = r.nextFloat();
			targets[i][0][0] = inputs[i][0][0] * 3.0f + 2.0f;
		}
	}
}