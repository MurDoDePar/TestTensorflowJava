package org.murdodepar.test.junit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.HashSet;
import java.util.Iterator;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.murdodepar.test.GraphUtil;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.test.TestUtil;

/** Unit tests for {@link org.murdodepar.test.GraphUtil}. */
@RunWith(JUnit4.class)
public class GraphUtilJUnit {

	@Test
	public void GraphUtilTest() {
		byte[] graphDef;
		// Create a graph for A * X + B
		try (GraphUtil gu = new GraphUtil()) {
			TestUtil.transpose_A_times_X(gu.getGraph(), new int[2][2]);
			graphDef = gu.getGraph().toGraphDef();
		}
		// Import the GraphDef and find all the nodes.
		try (GraphUtil gu = new GraphUtil()) {
			gu.getGraph().importGraphDef(graphDef);
			validateImportedGraph(gu.getGraph(), "");
		}
		try (GraphUtil gu = new GraphUtil()) {
			gu.getGraph().importGraphDef(graphDef, "BugsBunny");
			validateImportedGraph(gu.getGraph(), "BugsBunny/");
		}
	}
	// Helper function whose implementation is based on knowledge of how
	// TestUtil.transpose_A_times_X is implemented.
	private static void validateImportedGraph(Graph g, String prefix) {
		Operation op = g.operation(prefix + "A");
		assertNotNull(op);
		assertEquals(prefix + "A", op.name());
		assertEquals("Const", op.type());
		assertEquals(1, op.numOutputs());
		assertEquals(op, op.output(0).op());

		op = g.operation(prefix + "X");
		assertNotNull(op);
		assertEquals(prefix + "X", op.name());
		assertEquals("Placeholder", op.type());
		assertEquals(1, op.numOutputs());
		assertEquals(op, op.output(0).op());

		op = g.operation(prefix + "Y");
		assertNotNull(op);
		assertEquals(prefix + "Y", op.name());
		assertEquals("MatMul", op.type());
		assertEquals(1, op.numOutputs());
		assertEquals(op, op.output(0).op());
	}
	@Test
	public void iterateOverOperations() {
		try (GraphUtil gu = new GraphUtil()) {
			Iterator<Operation> iterator = gu.getGraph().operations();
			HashSet<Operation> operations;

			assertFalse(iterator.hasNext());

			operations = new HashSet<>();
			operations.add(TestUtil.constant(gu.getGraph(), "Const-A", Float.valueOf(1.0f)).op());
			operations.add(TestUtil.constant(gu.getGraph(), "Const-B", Integer.valueOf(23)).op());
			operations.add(TestUtil.constant(gu.getGraph(), "Const-C", Double.valueOf(1.618)).op());

			iterator = gu.getGraph().operations();

			assertTrue(iterator.hasNext());
			assertTrue(operations.remove(iterator.next()));

			assertTrue(iterator.hasNext());
			assertTrue(operations.remove(iterator.next()));

			assertTrue(iterator.hasNext());
			assertTrue(operations.remove(iterator.next()));

			assertFalse(iterator.hasNext());
		}
	}

	@Test
	public void failImportOnInvalidGraphDefs() {
		try (GraphUtil gu = new GraphUtil()) {
			try {
				gu.getGraph().importGraphDef(null);
			} catch (IllegalArgumentException e) {
				// expected exception.
			}

			try {
				gu.getGraph().importGraphDef(new byte[] {1});
			} catch (IllegalArgumentException e) {
				// expected exception.
			}
		}
	}

	@Test
	public void failOnUseAfterClose() {
		GraphUtil gu = new GraphUtil();
		gu.close();
		try {
			gu.getGraph().toGraphDef();
		} catch (IllegalStateException e) {
			// expected exception.
		}
	}
	@Test
	public void opBuilder() {
	    try (GraphUtil gu = new GraphUtil();) {
	          Output<OperationBuilder> c1 = gu.setConst("C1", DataType.INT32, new Integer(4));
	          Output<OperationBuilder> c2 = gu.setConst("C2", DataType.INT32, new Integer(3));
	          gu.setAdd("addC", c1, c2);
	          Tensor<?> cz = gu.getSession().runner().fetch("addC").run().get(0);
	          assertEquals(7, cz.intValue());

	          Output<OperationBuilder> p1 = gu.setPlaceholder("P1", DataType.INT32);
	          Output<OperationBuilder> p2 = gu.setPlaceholder("P2", DataType.INT32);
	          Tensor<Integer> input1 = Tensors.create(new Integer(4));
	          Tensor<Integer> input2 = Tensors.create(new Integer(3));
	          gu.setAdd("addP", p1, p2);
	          Tensor<?> pz = gu.getSession().runner()
	        		  .feed("P1", input1).feed("P2", input2)
	        		  .fetch("addP").run().get(0);
	          assertEquals(7, pz.intValue());

	          gu.setSub("subC", c1, c2);
	          cz = gu.getSession().runner().fetch("subC").run().get(0);
	          assertEquals(1, cz.intValue());

	          gu.setSub("subP", p1, p2);
	          pz = gu.getSession().runner()
	        		  .feed("P1", input1).feed("P2", input2)
	        		  .fetch("subP").run().get(0);
	          assertEquals(1, pz.intValue());

	          gu.setMul("mulC", c1, c2);
	          cz = gu.getSession().runner().fetch("mulC").run().get(0);
	          assertEquals(12, cz.intValue());

	          gu.setMul("mulP", p1, p2);
	          pz = gu.getSession().runner()
	        		  .feed("P1", input1).feed("P2", input2)
	        		  .fetch("mulP").run().get(0);
	          assertEquals(12, pz.intValue());

	          gu.setSquare("squareC", c1);
	          cz = gu.getSession().runner().fetch("squareC").run().get(0);
	          assertEquals(16, cz.intValue());

	          gu.setSquare("squareP", p1);
	          pz = gu.getSession().runner()
	        		  .feed("P1", input1)
	        		  .fetch("squareP").run().get(0);
	          assertEquals(16, pz.intValue());
	          
	    }
	}
}
