package org.tensorflow.test.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.op.Operands;
import org.tensorflow.test.TestUtil;

/** Unit tests for {@link org.tensorflow.op.Operands}. */
@RunWith(JUnit4.class)
public class OperandsTest {

  @Test
  public void createOutputArrayFromOperandList() {
    try (Graph g = new Graph()) {
      Operation split = TestUtil.split(g, "split", new int[] {0, 1, 2}, 3);
      List<Output<Integer>> list =
          Arrays.asList(split.<Integer>output(0), split.<Integer>output(2));
      Output<?>[] array = Operands.asOutputs(list);
      assertEquals(list.size(), array.length);
      assertSame(array[0], list.get(0));
      assertSame(array[1], list.get(1));
    }
  }
}