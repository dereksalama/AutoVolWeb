package edu.autovolweb;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.servlet.Servlet;
import javax.servlet.annotation.WebServlet;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Servlet implementation class AvgKnnClassifyServlet
 */
@WebServlet("/AvgKnnClassifyServlet")
public class AvgKnnClassifyServlet extends BaseKnnClassify implements Servlet {
	private static final long serialVersionUID = 1L;
	
	public static final int NUM_VECTORS_TO_AVG = 4;

	@Override
	protected Instances loadData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		Instances avgData = new Instances(data, data.numInstances() / 
				NUM_VECTORS_TO_AVG);
		LinkedList<Instance> queue = new LinkedList<>();
		for (Instance i : data) {
			queue.addFirst(i);
			if (queue.size() > NUM_VECTORS_TO_AVG) {
				queue.removeLast();
			} else if (queue.size() == NUM_VECTORS_TO_AVG) {
				Instance avg = DataUploadServlet.avgInstances(queue, avgData);
				avgData.add(avg);
			}
		}
		
		return avgData;
	}

	@Override
	protected Instance constructTarget(String input, String userId) {
		List<CurrentStateData> states = CurrentStateUtil.fromJson(input);
		
		List<Instance> instances = new ArrayList<Instance>();
		for (CurrentStateData state : states) {
			Instance target = CurrentStateUtil.toInstance(state);
			instances.add(target);
		}
		
		Instance avg = DataUploadServlet.avgInstances(instances, getDataset(userId));
		return avg;
	}


}
