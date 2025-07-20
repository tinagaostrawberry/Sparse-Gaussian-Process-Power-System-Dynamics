function cb_legend(~,evt)
% Allows clicking on legend to toggle line on/off

if strcmp(evt.Peer.Visible,'on')
    evt.Peer.Visible = 'off';
else 
    evt.Peer.Visible = 'on';
end
end