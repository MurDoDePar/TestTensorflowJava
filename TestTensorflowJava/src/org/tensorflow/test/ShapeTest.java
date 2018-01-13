package org.tensorflow.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Shape;

/** Unit tests for {@link Shape}. */
@RunWith(JUnit4.class)
public class ShapeTest {

  @Test
  public void unknown() {
    assertEquals(-1, Shape.unknown().numDimensions());
    assertEquals("<unknown>", Shape.unknown().toString());
  }

  @Test
  public void scalar() {
    assertEquals(0, Shape.scalar().numDimensions());
    assertEquals("[]", Shape.scalar().toString());
  }

  @Test
  public void make() {
    Shape s = Shape.make(2);
    assertEquals(1, s.numDimensions());
    assertEquals(2, s.size(0));
    assertEquals("[2]", s.toString());

    s = Shape.make(2, 3);
    assertEquals(2, s.numDimensions());
    assertEquals(2, s.size(0));
    assertEquals(3, s.size(1));
    assertEquals("[2, 3]", s.toString());

    s = Shape.make(-1, 2, 3);
    assertEquals(3, s.numDimensions());
    assertEquals(-1, s.size(0));
    assertEquals(2, s.size(1));
    assertEquals(3, s.size(2));
    assertEquals("[?, 2, 3]", s.toString());
  }

  @Test
  public void nodesInAGraph() {
    try (Graph g = new Graph()) {
      Output<Float> n = TestUtil.placeholder(g, "feed", Float.class);
      assertEquals(-1, n.shape().numDimensions());

      n = TestUtil.constant(g, "scalar", 3);
      assertEquals(0, n.shape().numDimensions());

      n = TestUtil.constant(g, "vector", new float[2]);
      assertEquals(1, n.shape().numDimensions());
      assertEquals(2, n.shape().size(0));

      n = TestUtil.constant(g, "matrix", new float[4][5]);
      assertEquals(2, n.shape().numDimensions());
      assertEquals(4, n.shape().size(0));
      assertEquals(5, n.shape().size(1));
    }
  }

  @Test
  public void equalsWorksCorrectly() {
    assertEquals(Shape.scalar(), Shape.scalar());
    assertEquals(Shape.make(1, 2, 3), Shape.make(1, 2, 3));

    assertNotEquals(Shape.make(1, 2), null);
    assertNotEquals(Shape.make(1, 2), new Object());
    assertNotEquals(Shape.make(1, 2, 3), Shape.make(1, 2, 4));

    assertNotEquals(Shape.unknown(), Shape.unknown());
    assertNotEquals(Shape.make(-1), Shape.make(-1));
    assertNotEquals(Shape.make(1, -1, 3), Shape.make(1, -1, 3));
  }

  @Test
  public void hashCodeIsAsExpected() {
    assertEquals(Shape.make(1, 2, 3, 4).hashCode(), Shape.make(1, 2, 3, 4).hashCode());
    assertEquals(Shape.scalar().hashCode(), Shape.scalar().hashCode());
    assertEquals(Shape.unknown().hashCode(), Shape.unknown().hashCode());

    assertNotEquals(Shape.make(1, 2).hashCode(), Shape.make(1, 3).hashCode());
  }
}