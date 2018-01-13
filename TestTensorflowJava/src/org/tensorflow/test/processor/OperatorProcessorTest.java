package org.tensorflow.test.processor;
/*
import static com.google.testing.compile.CompilationSubject.assertThat;

import com.google.testing.compile.Compilation;
import com.google.testing.compile.Compiler;
import com.google.testing.compile.JavaFileObjects;
*/
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Basic tests for {@link org.tensorflow.processor.operator.OperatorProcessor}. */
@RunWith(JUnit4.class)
public final class OperatorProcessorTest {

  /*@Test
  public void basicGood() {
    Compilation compile = compile("org/tensorflow/processor/operator/good/BasicGood.java");
    assertThat(compile).succeededWithoutWarnings();
    assertThat(compile).generatedSourceFile("org.tensorflow.op.Ops");
  }

  @Test
  public void basicBad() {
    assertThat(compile("org/tensorflow/processor/operator/bad/BasicBad.java")).failed();
  }

  // Create a compilation unit that includes the @Operator annotation and processor.
  private static Compilation compile(String path) {
    return Compiler.javac()
        .withProcessors(new OperatorProcessor())
        .compile(
            JavaFileObjects.forResource("src/main/java/org/tensorflow/op/annotation/Operator.java"),
            JavaFileObjects.forResource(path));
  }*/
}