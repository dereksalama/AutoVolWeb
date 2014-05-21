package edu.autovolweb;


import javax.servlet.annotation.WebServlet;

import weka.core.Instance;
import weka.core.Instances;

import com.google.gson.Gson;

/**
 * Servlet implementation class KnnClassifyServlet
 */
@WebServlet("/KnnClassifyServlet")
public class KnnClassifyServlet extends BaseKnnClassify {
	private static final long serialVersionUID = 1L;

	
	@Override
	protected Instances loadData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		return data;
	}
	
	@Override
	protected Instance constructTarget(String input, String userId) {
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);
		return CurrentStateUtil.toInstance(state);
	}
}
