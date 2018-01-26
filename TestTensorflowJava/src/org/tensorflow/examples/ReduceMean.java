package org.tensorflow.examples;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
/*
 * https://github.com/tensorflow/tensorflow/issues/8280
 */
public class ReduceMean {
  public static void main(String[] args) {
    try (Graph g = new Graph();
        Session s = new Session(g)) {
      // Build the graph
      Output placeholder =
          g.opBuilder("Placeholder", "input").setAttr("dtype", DataType.FLOAT).build().output(0);
      Output mean = reduce(g, "Mean", "output", placeholder);

      // Execute it
      try (Tensor in = Tensor.create(new float[]{1,2,3,4,5});
          Tensor out = s.runner().feed("input", in).fetch("output").run().get(0)) {
        System.out.println("Result: " + out.floatValue());
      }
    }
  }

  public static Output reduce(Graph g, String type, String name, Output input) {
    try (Tensor t = Tensor.create(new int[] {0})) {
      Output reductionIndices =
          g.opBuilder("Const", "ReductionIndices")
              .setAttr("dtype", t.dataType())
              .setAttr("value", t)
              .build()
              .output(0);
      return g.opBuilder(type, name)
          .setAttr("T", DataType.FLOAT)
          .setAttr("Tidx", DataType.INT32)
          .addInput(input)
          .addInput(reductionIndices)
          .build()
          .output(0);
    }
  }
}