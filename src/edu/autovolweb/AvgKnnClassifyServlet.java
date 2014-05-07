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
		Instances avgData = avgData(data, NUM_VECTORS_TO_AVG);
		
		return avgData;
	}
	
	public static Instances avgData(Instances data, int numVecsToAvg) {
		Instances avgData = new Instances(data, data.numInstances() / 
				numVecsToAvg);
		avgData.setClass(avgData.attribute("ringer"));
		LinkedList<Instance> queue = new LinkedList<>();
		double currentDay = 0.0;
		for (Instance i : data) {
			double day = i.value(data.attribute("day"));
			if (day != currentDay) {
				queue.clear();
				currentDay = day;
			}
			queue.addFirst(i);
			if (queue.size() > numVecsToAvg) {
				queue.removeLast();
			} else if (queue.size() < numVecsToAvg) {
				continue;
			}
			Instance avg = DataUploadServlet.avgInstances(queue, avgData);
			avgData.add(avg);
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
