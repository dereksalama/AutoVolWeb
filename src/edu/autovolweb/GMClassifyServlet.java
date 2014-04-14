package edu.autovolweb;

import javax.servlet.annotation.WebServlet;

import weka.core.Instance;

import com.google.gson.Gson;

/**
 * Servlet implementation class GMClassifyServlet
 */
@WebServlet("/GMClassifyServlet")
public class GMClassifyServlet extends BaseGMClassifyServlet {
	private static final long serialVersionUID = 1L;
	
	@Override
	protected String getEMModelFile() {
		return DataUploadServlet.EM_MODEL_FILE;
	}

	@Override
	protected String getClusterLabelsFile() {
		return DataUploadServlet.CLUSTER_LABELS_FILE;
	}

	@Override
	protected Instance createTarget(String input, String userId) throws Exception {
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);

		Instance locTarget = CurrentStateUtil.extractLocInstance(state);
		String locCluster = "" + ((int) getLocClusterer(userId).clusterInstance(locTarget));
		if (!getLocClusters(userId).contains(locCluster)) {
			locCluster = "other";
		}
		Instance target = CurrentStateUtil.toUnlabeledLocInstance(state,
				locCluster, getLocClusters(userId));
		
		return target;
	}

}
