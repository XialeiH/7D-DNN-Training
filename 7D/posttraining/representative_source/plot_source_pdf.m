function plot_source_pdf(out,maps,q,residual)
file=sprintf('orientation(%g)_contrast(%g)_TF(%g)_SF(%g).pdf',q);
names={'S','C','I','E'};
    f=figure('Visible','off','Color','w','Units','inches','Position',[0 0 23 19]);
    set(f,'PaperUnits','inches','PaperSize',[23 19],'PaperPosition',[0 0 23 19]);
    maps(:,:,4)=0.6923*maps(:,:,1)+0.3077*maps(:,:,2);
    for j=1:4
        left=2.3+10.4*mod(j-1,2); bottom=11-9.4*floor((j-1)/2);
        ax=axes(f,'Units','inches','Position',[left bottom 6.3 6.3]);
        z=maps(:,:,j); z=z([1:40 40],[1:40 40]);
        surface(ax,0.5:40.5,0.5:40.5,zeros(41),z,'FaceColor','flat','EdgeColor','none');
        view(ax,2); axis(ax,'ij'); axis(ax,'image');
        set(ax,'XLim',[0.5 40.5],'YLim',[0.5 40.5],'XTick',[1 10 20 30 40],...
            'YTick',[1 10 20 30 40],'FontName','Arial','FontSize',26,'Layer','top','Box','on');
        colormap(ax,jet(256));
        lo=min(maps(:,:,j),[],'all'); hi=max(maps(:,:,j),[],'all');
        if hi==lo, hi=lo+1; end
        clim(ax,[lo hi]);
        title(ax,names{j},'FontSize',30,'FontWeight','bold');
        xlabel(ax,'Pixel column','FontSize',26); ylabel(ax,'Pixel row','FontSize',26);
        cb=colorbar(ax); set(ax,'Units','inches','Position',[left bottom 6.3 6.3]);
        set(cb,'Units','inches','Position',[left+6.8 bottom 0.556 6.3],...
            'FontName','Arial','FontSize',26);
        title(cb,'sp/s','FontSize',26);
        drawnow;
        % Reserve 0.4 inches between tick-label extents and axis titles.
        ax.XLabel.Units='inches'; ax.XLabel.Position=[3.15 -0.9 0];
        ax.YLabel.Units='inches'; ax.YLabel.Position=[-1.1 3.15 0];
    end
    set(f,'Position',[0 0 23 19]);
    annotation(f,'textbox',[.1 .005 .8 .018],'String',sprintf('300 iterations; final step max change (sp/s) = %.3g',residual),'EdgeColor','none','HorizontalAlignment','center','FontSize',20);
    print(f,fullfile(out,file),'-dpdf','-painters','-r600');
    close(f);
end
