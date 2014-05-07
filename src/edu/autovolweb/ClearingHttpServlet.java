package edu.autovolweb;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;

public abstract class ClearingHttpServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
	private ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
	
	protected abstract void clear();
	
	@Override
	public void init() throws ServletException {
		super.init();
		executor.schedule(new Runnable() {
			
			@Override
			public void run() {
				clear();
			}
		}, 12, TimeUnit.HOURS);
	}
}
