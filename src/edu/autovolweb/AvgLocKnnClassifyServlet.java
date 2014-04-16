package edu.autovolweb;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.servlet.annotation.WebServlet;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Servlet implementation class AvgLocKnnClassifyServlet
 */
@WebServlet("/AvgLocKnnClassifyServlet")
public class AvgLocKnnClassifyServlet extends LocKnnClassifyServlet {
	private static final long serialVersionUID = 1L;
	

	@Override
	protected Instances loadData(String userId) throws Exception {
		Instances allDataLoc = super.loadData(userId);
		
		Instances avgData = new Instances(allDataLoc, allDataLoc.numInstances() / 
				AvgKnnClassifyServlet.NUM_VECTORS_TO_AVG);
		LinkedList<Instance> queue = new LinkedList<>();
		for (Instance i : allDataLoc) {
			queue.addFirst(i);
			if (queue.size() > AvgKnnClassifyServlet.NUM_VECTORS_TO_AVG) {
				queue.removeLast();
			} else if (queue.size() == AvgKnnClassifyServlet.NUM_VECTORS_TO_AVG) {
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
			Instance locTarget = CurrentStateUtil.extractLocInstance(state);
			String locCluster;
			try {
				locCluster = "" + ((int) getLocClusterer(userId).clusterInstance(locTarget));
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}
			if (!getLocClusters(userId).contains(locCluster)) {
				locCluster = "other";
			}
			
			Instance target = CurrentStateUtil.toLocInstance(state, locCluster, 
					getLocClusters(userId));
			instances.add(target);
		}
		
		Instance avg = DataUploadServlet.avgInstances(instances, getDataset(userId));
		return avg;
	}
	
}
