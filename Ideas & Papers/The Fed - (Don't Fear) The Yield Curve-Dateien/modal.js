pubwebApp.run(['$templateCache', function($templateCache) {
    $templateCache.put('modal-fullscreen.html', '<div id="modal" class="modal modal--fullscreen" tabindex="-1"><div class="modal__content"><div class="modal__header"><h4 class="modal__title"></h4><button type="button" title="close" aria-label="close" class="close" data-dismiss="modal" data-modal-button-close="">×</button></div><div class="modal__body"></div></div></div>');
}]);

pubwebApp.controller('fullscreenModalController', function($scope, $rootscope) {
    $scope.modalTitle = "";
    $scope.$modal = null;
})
    .directive("tablePopout", function($compile, $templateCache) {
    return {
        scope: false,
        link: function(scope, el, attrs) {
                var $table = $('table', el);
            var $modalButton = $('<a href="#" class="button-popout-launcher">Make Full Screen<span class="icon icon__sm icon--right icon-enter-fullscreen"></span></a>');
                $modalButton.insertBefore($table);

            $modalButton.on('click', function(event) {

                event.preventDefault();

                scope.$modal = $($templateCache.get('modal-fullscreen.html'));

                var $modalTitle = $('.modal__title', scope.$modal);

                $modalTitle.text($table.attr('title'));

                var $html = $(el.html());

                var $modalBody = $('.modal__body', scope.$modal);

                $('body').append(scope.$modal);

                angular.element($modalBody).injector().invoke(function($compile) {

                    var $scope = angular.element($modalBody).scope();
                    $modalBody.append($html);

                    $scope.$modal.find('button.close').on('click', function() {
                        pubwebApp.closeModal($scope);
                    });

                    
                    // the copy will contain the launcher button. remove it.
                    $modalBody.find('a.button-popout-launcher').remove();
                    
                    // use escape to remove  it
                    document.addEventListener('keydown',function(event){
                        if(event.key === 'Escape' || event.key === 'Esc'){
                            pubwebApp.closeModal($scope);
                        }
                    });

                    $compile($html)($scope);
                    // Finally, refresh the watch expressions in the new element
                    $scope.$apply();

                    $scope.$modal.focus();
                });
                $('body').addClass('modal-open');
                scope.$modal.addClass('on');
            });
        }
        };
});

pubwebApp.closeModal = function($scope){
    $scope.$modal.remove();
    $('body').removeClass('modal-open');
}