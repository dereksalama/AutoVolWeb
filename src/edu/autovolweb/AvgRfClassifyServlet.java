package edu.autovolweb;

import java.util.ArrayList;
import java.util.List;

import javax.servlet.annotation.WebServlet;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Servlet implementation class AvgRfClassifyServlet
 */
@WebServlet("/AvgRfClassifyServlet")
public class AvgRfClassifyServlet extends RfClassifyServlet {
	private static final long serialVersionUID = 1L;
	private static final int NUM_VECTORS_TO_AVG = 4;

    @Override
	protected Instances loadData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		Instances avgData = AvgKnnClassifyServlet.avgData(data, NUM_VECTORS_TO_AVG);
		return avgData;
	}
    
	@Override
	protected Instance constructTarget(String input, String userId) {
		List<CurrentStateData> states = CurrentStateUtil.fromJson(input);
		
		List<Instance> instances = new ArrayList<Instance>();
		for (int i = states.size() - NUM_VECTORS_TO_AVG; i < states.size(); i++) {
			CurrentStateData state = states.get(i);
			Instance target = CurrentStateUtil.toInstance(state);
			instances.add(target);
		}
		
		Instance avg = DataUploadServlet.avgInstances(instances, instances.get(0).dataset());
		return avg;
	}

}
