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

package com.domelis.test;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Example {
	public static void main(String[] args) {
		try (Graph g = new Graph(); Session s = new Session(g)) {
			// Construct a graph to add two float Tensors, using placeholders.
			Output<OperationBuilder> x = g.opBuilder("Placeholder", "x").setAttr("dtype", DataType.FLOAT).build().output(0);
			Output<OperationBuilder> y = g.opBuilder("Placeholder", "y").setAttr("dtype", DataType.FLOAT).build().output(0);
			Output<OperationBuilder> z = g.opBuilder("Add", "z").addInput(x).addInput(y).build().output(0);
			// Execute the graph multiple times, each time with a different value of x and y
			float[] X = new float[] { 1, 2, 3 };
			float[] Y = new float[] { 4, 5, 6 };
			for (int i = 0; i < X.length; i++) {
				try (
						Tensor<?> tx = Tensor.create(X[i]);
						Tensor<?> ty = Tensor.create(Y[i]);
						Tensor<?> tz = s.runner().feed("x", tx).feed("y", ty).fetch("z").run().get(0)
				) {
					System.out.println(X[i] + " + " + Y[i] + " = " + tz.floatValue());
				}
			}
		}
	}
}