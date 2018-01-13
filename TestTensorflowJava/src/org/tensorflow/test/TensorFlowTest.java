package org.tensorflow.test;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.TensorFlow;

/** Unit tests for {@link org.tensorflow.TensorFlow}. */
@RunWith(JUnit4.class)
public class TensorFlowTest {
  @Test
  public void version() {
    assertTrue(TensorFlow.version().length() > 0);
  }

  @Test
  public void registeredOpList() {
    // Would be nice to actually parse the output as a tensorflow.OpList protocol buffer message,
    // but as of May 2017, bazel support for generating Java code from protocol buffer definitions
    // was not sorted out. Revisit? Till then, at least exercise the code.
    assertTrue(TensorFlow.registeredOpList().length > 0);
  }

  @Test
  public void loadLibrary() {
    // TODO(ashankar): This tell will fail when built with --config=monolithic.
    // Figure out how we can ignore the test in that case.
    try (Graph g = new Graph()) {
      // Build a graph with an unrecognized operation.
      try {
        g.opBuilder("MyTest", "MyTest").build();
        fail("should not be able to construct graphs with unregistered ops");
      } catch (IllegalArgumentException e) {
        // expected exception
      }

      // Load the library containing the operation.
      byte[] opList = TensorFlow.loadLibrary("tensorflow/java/my_test_op.so");
      assertTrue(opList.length > 0);

      // Now graph building should succeed.
      g.opBuilder("MyTest", "MyTest").build();
    }
  }
}