package org.tensorflow.test;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.SavedModelBundle;

/** Unit tests for {@link org.tensorflow.SavedModelBundle}. */
@RunWith(JUnit4.class)
public class SavedModelBundleTest {

  private static final String SAVED_MODEL_PATH =
      "tensorflow/cc/saved_model/testdata/half_plus_two/00000123";

  @Test
  public void load() {
    try (SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PATH, "serve")) {
      assertNotNull(bundle.session());
      assertNotNull(bundle.graph());
      assertNotNull(bundle.metaGraphDef());
    }
  }

  @Test
  public void loadNonExistentBundle() {
    try {
      SavedModelBundle bundle = SavedModelBundle.load("__BAD__", "serve");
      bundle.close();
      fail("not expected");
    } catch (org.tensorflow.TensorFlowException e) {
      // expected exception
      assertTrue(e.getMessage().contains("SavedModel not found"));
    }
  }
}