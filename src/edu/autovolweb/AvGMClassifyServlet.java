package edu.autovolweb;

import java.util.ArrayList;
import java.util.List;

import javax.servlet.annotation.WebServlet;

import weka.core.Instance;

/**
 * Servlet implementation class AvGMClassifyServlet
 */
@WebServlet("/AvGMClassifyServlet")
public class AvGMClassifyServlet extends BaseGMClassifyServlet {
	private static final long serialVersionUID = 1L;

	@Override
	protected String getEMModelFile() {
		return DataUploadServlet.AVG_EM_MODEL_FILE;
	}

	@Override
	protected String getClusterLabelsFile() {
		return DataUploadServlet.AVG_CLUSTER_LABELS_FILE;
	}

	@Override
	protected Instance createTarget(String input, String userId)
			throws Exception {
		List<CurrentStateData> states = CurrentStateUtil.fromJson(input);
		
		List<Instance> instances = new ArrayList<Instance>();
		for (CurrentStateData state : states) {
			Instance locTarget = CurrentStateUtil.extractLocInstance(state);
			String locCluster = "" + ((int) getLocClusterer(userId).clusterInstance(locTarget));
			if (!getLocClusters(userId).contains(locCluster)) {
				locCluster = "other";
			}
			Instance target = CurrentStateUtil.toUnlabeledLocInstance(state,
					locCluster, getLocClusters(userId));
			
			instances.add(target);
		}
		
		Instance avg = DataUploadServlet.avgInstances(instances, 
				CurrentStateUtil.createUnlabeledLocDataset(getLocClusters(userId)));
		
		return avg;
	}


}
